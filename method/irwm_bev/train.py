"""Eight-rank training with route-macro validation and complete resumable checkpoints."""
import argparse
from contextlib import nullcontext
from datetime import datetime,timezone,timedelta
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import time
import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler,BatchSampler,Subset
from method.full_diffusion.garage import FrozenGaragePerception
from .data import SequenceDataset,TeacherCache,collate
from .geometry import align_to_anchor,path_corridor
from .model import ActionBEVWorldModel
from .loss import error_metrics,training_loss
from .launch import guard_context,ROOT,AUTHORIZED,write_json

STOP=False


def request_stop(*_):
    global STOP
    STOP=True


def rng_state():
    return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),
            'numpy':np.random.get_state(),'python':random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def atomic_save(value,path):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:
        torch.save(value,f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary,path)


def save_checkpoint(output,model,optimizer,scheduler,progress,config,manifest,metrics=None,epoch_file=False,best=False):
    states=[None]*dist.get_world_size()
    dist.all_gather_object(states,rng_state())
    if dist.get_rank()==0:
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
               'rng_by_rank':states,'progress':progress,'config':config,'dataset':manifest,'metrics':metrics,
               'precision':'bf16 autocast; fp32 parameters/optimizer; no GradScaler',
               'created_utc':datetime.now(timezone.utc).isoformat()}
        latest=output/'latest.pt'
        if latest.exists():
            prior=output/'latest.previous.pt.tmp'
            if prior.exists():
                prior.unlink()
            os.link(latest,prior)
            os.replace(prior,output/'latest.previous.pt')
        atomic_save(state,latest)
        if epoch_file:
            weights={k:v for k,v in state.items() if k not in ('optimizer','scheduler','rng_by_rank')}
            atomic_save(weights,output/f"epoch_{progress['epoch']:04}.pt")
        if best:
            temp=output/'best.pt.tmp'
            if temp.exists():
                temp.unlink()
            os.link(latest,temp)
            os.replace(temp,output/'best.pt')
        write_json(output/'checkpoint.json',{'progress':progress,'metrics':metrics,'full_latest':str(latest),
                                            'best_updated':best,'saved_utc':state['created_utc']})
    dist.barrier()


def load_checkpoint(path,model,optimizer,scheduler,config,manifest):
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    if ckpt['config']!=config or ckpt['dataset']!=manifest:
        raise RuntimeError('Resume config/split/teacher differs from checkpoint')
    if len(ckpt['rng_by_rank'])!=dist.get_world_size():
        raise RuntimeError('Resume requires original world size')
    model.load_state_dict(ckpt['model'],strict=True)
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    restore_rng(ckpt['rng_by_rank'][dist.get_rank()])
    return ckpt['progress']


class RemainingBatches:
    def __init__(self,batches,offset):
        self.batches=batches
        self.offset=offset
    def __len__(self):
        return max(0,len(self.batches)-self.offset)
    def __iter__(self):
        return itertools.islice(self.batches,self.offset,None)


def loader_for(dataset,config,workers=None,**kwargs):
    workers=config['workers_per_rank'] if workers is None else workers
    return DataLoader(dataset,num_workers=workers,pin_memory=True,collate_fn=collate,
                      **({'multiprocessing_context':'spawn','prefetch_factor':1,'timeout':120} if workers else {}),**kwargs)


@torch.no_grad()
def prepare(batch,teacher,device):
    features=teacher(batch['sensors'])
    batch={k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items() if k!='sensors'}
    b,n,c,h,w=features.shape
    aligned,valid=align_to_anchor(features.flatten(0,1),batch['source_in_anchor'].flatten(0,1))
    aligned=aligned.reshape(b,n,c,h,w)
    valid=valid.reshape(b,n,1,h,w)
    # Future images and future rotations terminate here, only creating target tensors.
    inputs={k:batch[k] for k in ('path','path_times','path_valid','query_xy','query_times','query_valid','speed','target_speed')}
    inputs.update(history=aligned[:,:3],history_valid=valid[:,:3])
    corridor=path_corridor(batch['path'],batch['path_valid'],batch['query_xy'],batch['query_valid'])
    return inputs,aligned[:,3:],valid[:,3:],corridor,batch['route_id']


def update(prepared,policy,optimizer,scheduler):
    inputs,target,mask,corridor,_=prepared
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        predictions=policy(**inputs)
    metrics=error_metrics(predictions,target,mask,inputs['query_valid'],corridor)
    loss=training_loss(metrics,inputs['query_valid'])
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite feature prediction loss')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return loss.detach(),norm.detach()


@torch.no_grad()
def validate(dataset,model,teacher,config,device,smoke=False,output=None,epoch=None):
    rank,world=dist.get_rank(),dist.get_world_size()
    selected=list(range(len(dataset)))
    if smoke:
        first={}
        for i,(rid,_) in enumerate(dataset.samples):
            first.setdefault(int(rid),i)
        selected=list(first.values())
    # Strided partition avoids duplicated/padded validation examples.
    # The fixed dev set is small. Avoid fresh process spawning at epoch boundaries.
    loader=loader_for(Subset(dataset,selected[rank::world]),config,workers=0,batch_size=config['batch_per_gpu'])
    if output is not None:
        write_json(output/f'rank{rank}.stage.json',{'phase':'validation','epoch':epoch,'batch':0,
                   'batches':len(loader),'utc':datetime.now(timezone.utc).isoformat()})
    names=('nmse','mse','corridor_nmse','cosine')
    totals=torch.zeros(len(dataset.routes),9,device=device,dtype=torch.float64)
    query_count=len(dataset.manifest['query_progress_m'])
    steps=torch.zeros(query_count,9,device=device,dtype=torch.float64)
    model.eval()
    for batch_id,batch in enumerate(loader):
        inputs,target,mask,corridor,rids=prepare(batch,teacher,device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            prediction=model(**inputs)
        measured=error_metrics(prediction,target,mask,inputs['query_valid'],corridor)
        baseline=error_metrics(inputs['history'][:,-1,None].expand_as(target),target,mask,inputs['query_valid'],corridor)
        valid=inputs['query_valid'].double()
        values=torch.stack([measured[n] for n in names]+[baseline[n] for n in names],-1).double()
        sequence=(values*valid[...,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1.)
        totals.index_add_(0,rids,torch.cat((sequence,torch.ones(len(rids),1,device=device,dtype=torch.float64)),1))
        steps[:,:8]+=(values*valid[...,None]).sum(0)
        steps[:,8]+=valid.sum(0)
        if output is not None and ((batch_id+1)%10==0 or batch_id+1==len(loader)):
            write_json(output/f'rank{rank}.stage.json',{'phase':'validation','epoch':epoch,'batch':batch_id+1,
                       'batches':len(loader),'utc':datetime.now(timezone.utc).isoformat()})
    dist.all_reduce(totals)
    dist.all_reduce(steps)
    present=totals[:,8]>0
    if int(present.sum())!=44:
        raise AssertionError(f'Expected all 44 dev routes, got {int(present.sum())}')
    route_means=totals[present,:8]/totals[present,8,None]
    macro=route_means.mean(0).cpu().tolist()
    result={'routes':44,'sequences':int(totals[:,8].sum()),'smoke_only':smoke,
            'model':dict(zip(names,macro[:4])),'copy_current':dict(zip(names,macro[4:])),
            'by_query':[{**dict(zip(names,(steps[k,:4]/steps[k,8].clamp_min(1.)).cpu().tolist())),
                         'valid_queries':int(steps[k,8])} for k in range(query_count)],
            'per_route':{dataset.routes[i]:{**dict(zip(names,(totals[i,:4]/totals[i,8]).cpu().tolist())),
                                          'sequences':int(totals[i,8])} for i in present.nonzero().flatten().tolist()}}
    if not math.isfinite(result['model']['nmse']):
        raise FloatingPointError('Nonfinite validation metric')
    model.train()
    return result


def acceptance(args,config,manifest,train,val,teacher,model,policy,optimizer,scheduler,device):
    from .tests import geometry_tests
    geometry_tests(device)
    rank,world=dist.get_rank(),dist.get_world_size()
    samples=[train[(rank*4+i)*13] for i in range(4)]
    prepared=prepare(collate(samples),teacher,device)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    losses=[]
    start=time.monotonic()
    for _ in range(2):
        loss,_=update(prepared,policy,optimizer,scheduler)
        losses.append(float(loss))
    progress={'epoch':0,'next_batch':2,'updates':2,'best_nmse':None}
    save_checkpoint(args.output,model,optimizer,scheduler,progress,config,manifest)
    update(prepared,policy,optimizer,scheduler)
    expected={n:p.detach().clone() for n,p in model.named_parameters()}
    load_checkpoint(args.output/'latest.pt',model,optimizer,scheduler,config,manifest)
    update(prepared,policy,optimizer,scheduler)
    difference=max(float((p-expected[n]).abs().max()) for n,p in model.named_parameters())
    changed=max(float((p-before[n]).abs().max()) for n,p in model.named_parameters())
    if difference>2e-6 or changed==0:
        raise AssertionError(f'Update/resume mismatch: {difference}, change={changed}')
    if any(p.grad is not None or p.requires_grad for p in teacher.perception.parameters()):
        raise AssertionError('Teacher is not fully frozen')
    checksum=sum(p.double().sum() for p in model.parameters())
    checks=[torch.zeros_like(checksum) for _ in range(world)]
    dist.all_gather(checks,checksum)
    if max(float(abs(c-checks[0])) for c in checks)>1e-6:
        raise AssertionError('DDP replicas differ')
    inputs=prepared[0]
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        output=model(**inputs)
        changed_input={**inputs,'query_xy':inputs['query_xy']+torch.tensor([0.,2.],device=device)}
        intervention=model(**changed_input)
        action_effect=float((output-intervention).abs().max())
        arrival=torch.arange(1,11,device=device,dtype=torch.float32)[None].expand(4,-1)*.5
        ten=model.rollout(inputs['history'],inputs['history_valid'],inputs['path'],inputs['speed'],inputs['target_speed'],arrival)
    if action_effect==0 or ten.shape!=(4,10,64,64,64) or not torch.isfinite(ten).all():
        raise AssertionError('Action conditioning / ten-node inference failed')
    # Compare the exact same frames loaded fresh vs. from persisted bf16 cache.
    again=prepare(collate([train[(rank*4+i)*13] for i in range(4)]),teacher,device)
    cache_difference=float((prepared[1]-again[1]).abs().max())
    if cache_difference!=0:
        raise AssertionError(f'Feature cache changed target: {cache_difference}')
    metrics=validate(val,model,teacher,config,device,smoke=True)
    report={'passed':True,'rank':rank,'world_size':world,'hostname':socket.gethostname(),
            'optimizer_updates':3,'losses':losses,'parameter_change':changed,'resume_max_error':difference,
            'cache_max_error':cache_difference,'action_intervention_max_change':action_effect,
            'ten_node_output_shape':list(ten.shape),'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
            'seconds':time.monotonic()-start,'validation_routes':metrics['routes']}
    write_json(args.output/f'acceptance_rank{rank}.json',report)
    if rank==0:
        write_json(args.output/'validation_smoke.json',metrics)
        print(json.dumps({'event':'acceptance_passed',**report}),flush=True)
    dist.barrier()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=('acceptance','train'),required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    host=socket.gethostname().split('.')[0]
    job=next((j for j,h in AUTHORIZED.items() if h==host),None)
    guard_context(job)
    config=json.loads(args.config.read_text())
    rank,local,world=(int(os.environ[k]) for k in ('RANK','LOCAL_RANK','WORLD_SIZE'))
    if world!=8 or config['batch_per_gpu']!=4:
        raise RuntimeError('This run requires eight ranks and batch size four per rank')
    torch.set_num_threads(1)
    torch.cuda.set_device(local)
    device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(minutes=15))
    probe=torch.tensor(float(rank),device=device)
    dist.all_reduce(probe)
    assert float(probe)==28.
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    seed=config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    train=SequenceDataset(ROOT/config['index'], 'train',ROOT/config['cache'])
    val=SequenceDataset(ROOT/config['index'], 'val',ROOT/config['cache'])
    manifest=train.manifest
    perception=FrozenGaragePerception().to(device)
    teacher=TeacherCache(perception,device,chunk=config['teacher_chunk'])
    model=ActionBEVWorldModel(**config['model']).to(device)
    policy=DDP(model,device_ids=[local],broadcast_buffers=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    sampler=DistributedSampler(train,num_replicas=world,rank=rank,seed=seed,drop_last=True)
    batches=BatchSampler(sampler,config['batch_per_gpu'],drop_last=True)
    updates_per_epoch=len(batches)
    total=updates_per_epoch*config['epochs']
    warmup=config['warmup_updates']
    def factor(step):
        if step<warmup:
            return (step+1)/warmup
        return .1+.9*.5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,total-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    torch.manual_seed(seed+rank)
    np.random.seed(seed+rank)
    random.seed(seed+rank)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.phase=='acceptance':
        acceptance(args,config,manifest,train,val,teacher,model,policy,optimizer,scheduler,device)
        dist.destroy_process_group()
        return
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    progress={'epoch':0,'next_batch':0,'updates':0,'best_nmse':None}
    if args.resume:
        progress=load_checkpoint(args.output/'latest.pt',model,optimizer,scheduler,config,manifest)
    elif (args.output/'latest.pt').exists():
        raise FileExistsError('Existing training checkpoint requires --resume')
    if rank==0:
        write_json(args.output/'config.json',config)
        write_json(args.output/'dataset.json',manifest)
        code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
        write_json(args.output/'code_sha256.json',code)
        print(json.dumps({'event':'initialized','trainable_parameters':sum(p.numel() for p in model.parameters()),
                          'updates_per_epoch':updates_per_epoch,'global_batch':32,'epochs':config['epochs']}),flush=True)
    start=time.monotonic()
    interval=start
    interval_steps=0
    loss_sum=torch.zeros((),device=device)
    start_updates=progress['updates']
    for epoch in range(progress['epoch'],config['epochs']):
        write_json(args.output/f'rank{rank}.stage.json',{'phase':'training','epoch':epoch+1,
                   'utc':datetime.now(timezone.utc).isoformat()})
        sampler.set_epoch(epoch)
        offset=progress['next_batch'] if epoch==progress['epoch'] else 0
        loader=loader_for(train,config,batch_sampler=RemainingBatches(batches,offset))
        model.train()
        for i,batch in enumerate(loader,start=offset):
            prepared=prepare(batch,teacher,device)
            loss,norm=update(prepared,policy,optimizer,scheduler)
            progress.update(epoch=epoch,next_batch=i+1,updates=progress['updates']+1)
            loss_sum+=loss
            interval_steps+=1
            stopping=torch.tensor(int(STOP or (args.output/'STOP').exists()),device=device)
            dist.all_reduce(stopping,op=dist.ReduceOp.MAX)
            if progress['updates']==1 or progress['updates']%config['log_every']==0:
                dist.all_reduce(loss_sum)
                now=time.monotonic()
                record={'epoch':epoch+1,'batch':i+1,'updates':progress['updates'],'updates_per_epoch':updates_per_epoch,
                        'loss':float(loss_sum)/(interval_steps*world),'global_batch':32,
                        'lr':optimizer.param_groups[0]['lr'],'gradient_norm_rank0':float(norm),
                        'interval_seconds_per_update':(now-interval)/interval_steps,
                        'elapsed_seconds':now-start,'samples_per_second':(progress['updates']-start_updates)*32/(now-start),
                        'peak_gpu_gib_rank0':torch.cuda.max_memory_allocated()/2**30,
                        'cache_hits_rank0':teacher.hits,'cache_misses_rank0':teacher.misses,
                        'utc':datetime.now(timezone.utc).isoformat()}
                write_json(args.output/f'rank{rank}.progress.json',{'rank':rank,'host':host,'job':job,
                           'updates':progress['updates'],'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
                           'utc':record['utc']})
                if rank==0:
                    write_json(args.output/'progress.json',record)
                    with (args.output/'metrics.jsonl').open('a') as f:
                        f.write(json.dumps(record)+'\n')
                    print(json.dumps(record),flush=True)
                loss_sum.zero_()
                interval=now
                interval_steps=0
            if progress['updates']==1 or progress['updates']%config['checkpoint_every']==0 or int(stopping):
                save_checkpoint(args.output,model,optimizer,scheduler,progress,config,manifest)
            if int(stopping):
                if rank==0:
                    write_json(args.output/'stopped.json',progress)
                dist.destroy_process_group()
                return
        # Preserve every completed optimizer update before any validation I/O.
        save_checkpoint(args.output,model,optimizer,scheduler,progress,config,manifest)
        metrics=validate(val,model,teacher,config,device,output=args.output,epoch=epoch+1)
        score=metrics['model']['nmse']
        best=progress['best_nmse'] is None or score<progress['best_nmse']
        progress.update(epoch=epoch+1,next_batch=0)
        if best:
            progress['best_nmse']=score
        save_checkpoint(args.output,model,optimizer,scheduler,progress,config,manifest,metrics,epoch_file=True,best=best)
        if rank==0:
            write_json(args.output/f'validation_epoch_{epoch+1:04}.json',metrics)
            with (args.output/'validation.jsonl').open('a') as f:
                f.write(json.dumps({'epoch':epoch+1,'best':best,**{k:v for k,v in metrics.items() if k!='per_route'}})+'\n')
            print(json.dumps({'event':'epoch_complete','epoch':epoch+1,'validation':metrics['model'],
                              'copy_current':metrics['copy_current'],'best':best}),flush=True)
        dist.barrier()
    if rank==0:
        write_json(args.output/'completed.json',progress)
    dist.destroy_process_group()


if __name__=='__main__':
    main()
