"""Batch-one latency of trained timed BEV WM and the actual 100-step V3 policy."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
import numpy as np
import torch
from .launch import ROOT, guard_context, write_json, memory_snapshot


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def prepare_inputs(output):
    from .data import SequenceDataset
    from method.full_diffusion.data import GarageGeoDataset
    config=json.loads((ROOT/'method/irwm_bev_t/config.json').read_text())
    data=SequenceDataset(config['index'],'val',config['cache'])
    raw=GarageGeoDataset(ROOT/'data/full_diffusion/pdm_lite_seed901','val',training=False)
    samples=[]
    for index in np.linspace(0,len(data)-1,4).astype(int):
        rid,anchor=map(int,data.samples[index])
        meta=data.metadata(rid)
        frame=int(meta['frames'][anchor])
        route=Path(data.manifest['data_root'])/data.routes[rid]
        batch=raw.load_frame(route,frame,random.Random(901))
        batch={k:v[None] for k,v in batch.items() if k in ('rgb','lidar','speed','command','target_point')}
        features=torch.stack([data.sensor(rid,int(meta['frames'][i]))['bev'] for i in range(anchor-2,anchor+1)])[None]
        poses=meta['poses'][anchor-2:anchor+1].copy()
        samples.append({'batch':batch,'raw_bev_history':features,'poses':poses,'sample_id':str(route/f'{frame:04}')})
    torch.save(samples,output/'real_inputs.pt')
    write_json(output/'inputs.json',{'samples':[x['sample_id'] for x in samples],
                                    'shapes':{k:list(v.shape) for k,v in samples[0]['batch'].items()},
                                    'feature_shape':list(samples[0]['raw_bev_history'].shape)})
    print(json.dumps({'event':'real_inputs_ready','count':len(samples)}),flush=True)


@torch.no_grad()
def measure(args):
    guard_context('13053006')
    existing=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if existing:raise RuntimeError('GPU is in use; isolated inference measurement requires training to pause')
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    device=torch.device('cuda:0')
    # Match the deployed CARLA sensor agent; do not compile or change denoising.
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.deterministic=False
    from .model import ActionBEVWorldModel
    from .geometry import align_to_anchor
    from method.full_diffusion.garage import FrozenGaragePerception
    from method.full_diffusion2.data import frame_pose
    from method.full_diffusionv3.inference.policy import FullDiffusionV3Network
    world_checkpoint=torch.load(args.world_checkpoint,map_location='cpu',weights_only=False)
    world=ActionBEVWorldModel(**world_checkpoint['config']['model']).to(device).eval()
    world.load_state_dict(world_checkpoint['model'],strict=True)
    perception=FrozenGaragePerception().to(device).eval()
    policy=FullDiffusionV3Network(perception.net,args.v3_checkpoint,device).eval()
    assert policy.run_config['diffusion']['inference_steps']==100
    assert not policy.config.two_tp_input
    data=torch.load(args.output/'real_inputs.pt',map_location='cpu',weights_only=False)
    samples=[]
    for item in data:
        batch={k:v.to(device) for k,v in item['batch'].items()}
        raw=item['raw_bev_history'].to(device)
        poses=item['poses']
        matrices=[frame_pose({'theta':float(p[2]),'pos_global':p[:2]}) for p in poses]
        transforms=np.stack([np.linalg.solve(matrices[-1],m) for m in matrices])
        relative=np.stack([np.array([m[0,2],m[1,2],np.arctan2(m[1,0],m[0,0])]) for m in transforms]).astype(np.float32)
        relative=torch.tensor(relative,device=device)
        aligned,valid=align_to_anchor(raw.flatten(0,1),relative)
        sample={'batch':batch,'raw_history':raw,'relative':relative,'aligned':aligned[None],'valid':valid[None],
                'poses':poses,'matrices':matrices,'transforms':torch.tensor(transforms[:2],device=device,dtype=torch.float32)[None]}
        samples.append(sample)

    def seed_history(s):
        policy.history.entries.clear()
        policy.history.entries.append((-.5,s['raw_history'][:,0],s['matrices'][0]))
        policy.history.entries.append((-.25,s['raw_history'][:,1],s['matrices'][1]))
        policy.set_observation(s['poses'][-1],0.)

    def full(s):
        b=s['batch']
        return policy(rgb=b['rgb'],lidar_bev=b['lidar'],target_point=b['target_point'],
                      ego_vel=b['speed'],command=b['command'])

    # Prepare shared current features and action inputs outside timed regions.
    for s in samples:
        seed_history(s)
        full(s)
        s['delta']=policy.last_temporal_delta.clone()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            s['perception']=policy.perception(s['batch'])
        s['condition_args']=(*s['perception'],s['batch']['target_point'],s['raw_history'][:,:2],
                             s['transforms'],torch.ones((1,2),device=device,dtype=torch.bool),
                             s['batch']['speed'],s['batch']['command'])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred=world.rollout(s['aligned'],s['valid'],s['delta'],s['batch']['speed'])
        assert pred.shape==(1,6,64,64,64) and torch.isfinite(pred).all()
    report={'batch_size':1,'device':torch.cuda.get_device_name(0),'host':subprocess.check_output(['hostname'],text=True).strip(),
            'cpu_affinity':sorted(os.sched_getaffinity(0)),'torch_threads':torch.get_num_threads(),
            'torch_version':torch.__version__,'precision':'BF16 autocast / FP32 weights','compiled':False,
            'denoising_steps':100,'world_horizon_seconds':[.5,1,1.5,2,2.5,3],
            'world_parameters':sum(p.numel() for p in world.parameters()),
            'v3_head_parameters':sum(p.numel() for p in policy.head.parameters()),
            'backbone_and_aux_parameters':sum(p.numel() for p in policy.perception.parameters()),
            'world_checkpoint':str(args.world_checkpoint),'world_progress':world_checkpoint['progress'],
            'v3_checkpoint':str(args.v3_checkpoint),'v3_checkpoint_sha256':sha(args.v3_checkpoint),
            'world_checkpoint_sha256':sha(args.world_checkpoint),'timings':{},
            'excludes':'disk reads, checkpoint loading, CARLA rendering/RPC, raw sensor preprocessing; sensors/features preloaded on GPU',
            'history':'two cached real previous BEVs, never recompute historical backbone in timed loop',
            'action':'actual decoded V3 temporal_delta on the same observation',
            'utc':datetime.now(timezone.utc).isoformat()}

    def timed(name,fn,count,warmup,setup=None):
        for i in range(warmup):
            s=samples[i%len(samples)]
            if setup:setup(s)
            fn(s)
        torch.cuda.synchronize()
        wall=[];gpu=[]
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        for i in range(count):
            s=samples[i%len(samples)]
            if setup:setup(s)
            torch.cuda.synchronize()
            begin=time.perf_counter()
            start.record()
            result=fn(s)
            end.record();end.synchronize()
            wall.append((time.perf_counter()-begin)*1000)
            gpu.append(start.elapsed_time(end))
        values={'repetitions':count,'warmup':warmup,'mean_ms':statistics.mean(wall),
                'p50_ms':statistics.median(wall),'p95_ms':float(np.quantile(wall,.95)),
                'min_ms':min(wall),'max_ms':max(wall),'cuda_stream_mean_ms':statistics.mean(gpu)}
        report['timings'][name]=values
        write_json(args.output/'latency.json',report)
        print(json.dumps({'event':'timing','component':name,**values}),flush=True)
        return result

    def wm_core(s):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return world.rollout(s['aligned'],s['valid'],s['delta'],s['batch']['speed'])

    def wm_align(s):
        history,valid=align_to_anchor(s['raw_history'].flatten(0,1),s['relative'])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return world.rollout(history[None],valid[None],s['delta'],s['batch']['speed'])

    def perception_fn(s):
        with torch.autocast('cuda',dtype=torch.bfloat16):return policy.perception(s['batch'])

    def condition_fn(s):
        with torch.autocast('cuda',dtype=torch.bfloat16):return policy.head.prepare_condition(*s['condition_args'])

    def denoise_fn(s):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=policy.diffusion.sample(lambda noisy,t:policy.head(noisy,t,s['condition'])['predictions'][-1],
                                        1,device,policy.generator,100)
            return policy.head.codec.decode(out)

    def head_fn(s):
        s['condition']=condition_fn(s)
        return denoise_fn(s)

    for s in samples:s['condition']=condition_fn(s)
    timed('world_six_future_bevs_only',wm_core,120,20)
    timed('world_with_history_alignment',wm_align,120,20)
    timed('tfpp_current_perception_and_context',perception_fn,100,15)
    timed('v3_condition_preparation',condition_fn,100,15)
    timed('v3_100_step_denoising_decode',denoise_fn,40,8)
    timed('v3_condition_plus_100_step_head',head_fn,40,8)
    timed('v3_actual_policy_full_forward',full,60,10,seed_history)
    # Repeat the short WM benchmark after the longer planner timing to check drift.
    timed('world_six_future_bevs_repeat',wm_core,120,20)
    report.update(peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,memory=memory_snapshot(),completed=True)
    write_json(args.output/'latency.json',report)
    print(json.dumps({'event':'latency_complete','output':str(args.output/'latency.json')}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=('prepare','measure'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--world-checkpoint',type=Path)
    p.add_argument('--v3-checkpoint',type=Path)
    args=p.parse_args()
    args.output=args.output.resolve()
    allowed=Path('/fs/scratch/projects/common_ai/nix3sgh_cache/WMMPC').resolve()
    if not args.output.is_relative_to(allowed):raise ValueError('Output outside authorized cache')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    if args.phase=='prepare':prepare_inputs(args.output)
    else:measure(args)


if __name__=='__main__':main()
