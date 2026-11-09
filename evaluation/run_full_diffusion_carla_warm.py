"""One owned CARLA renderer, fresh evaluator/Python state for every route."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.bench2drive import check_job
from evaluation.fast_carla_io import cooperative_pause,wait_for_server
from evaluation.run_full_diffusion_carla import read,save,resources
from evaluation.run_full_diffusion_carla_fast import batch_trial,stop_evaluator


class OwnedServer:
    def __init__(self,args,gpu_uuid,directory):
        self.directory=directory;directory.mkdir(parents=True)
        (directory/'tmp').mkdir()
        cache=ROOT/'.runtime/carla_cache'/args.job_id/gpu_uuid
        for name in ('home','cache','unreal_user'):(cache/name).mkdir(parents=True,exist_ok=True)
        port=2000+args.slot*10
        binary=args.carla_root/'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping'
        command=[str(binary),'CarlaUE4','-RenderOffScreen','-nosound',f'-carla-rpc-port={port}',
                 f'-graphicsadapter={args.gpu}',f'-abslog={directory/"CarlaUE4.log"}',
                 f'-userdir={cache/"unreal_user"}','-unattended','-nosplash']
        env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':gpu_uuid,'CUDA_DEVICE_ORDER':'PCI_BUS_ID',
             'HOME':str(cache/'home'),'XDG_CACHE_HOME':str(cache/'cache'),'OMP_NUM_THREADS':'1',
             'TMPDIR':str(directory/'tmp'),
             'VK_ICD_FILENAMES':'/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json'})
        started=time.time()
        with (directory/'server_stdout.log').open('wb') as log:
            self.process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        self.record={'pid':self.process.pid,'port':port,'gpu_uuid':gpu_uuid,'job_id':args.job_id,
                     'command':command,'unix':started,'inference_steps':100}
        self.owner=directory/'server_process.json';save(self.owner,self.record)
        # CARLA's installed wheel is in the isolated evaluation environment.
        try:wait_for_server(self.process,'localhost',port,directory,timeout=args.startup_timeout)
        except BaseException:
            self.close();raise
        os.environ.update({'WMMPC_EXTERNAL_CARLA_PID':str(self.process.pid),
                           'WMMPC_EXTERNAL_CARLA_PORT':str(port),'WMMPC_EXTERNAL_CARLA_OWNER':str(self.owner),
                           'WMMPC_CACHE_STATIC_ROADS':'0'})

    def close(self):
        expected_stop=self.process.poll() is None
        if expected_stop:
            os.killpg(self.process.pid,signal.SIGTERM)
            try:self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid,signal.SIGKILL);self.process.wait(timeout=5)
        save(self.directory/'server_exit.json',{'pid':self.process.pid,'returncode':self.process.returncode,
                                               'unix':time.time(),'owner_finishing':expected_stop})
        for name in ('WMMPC_EXTERNAL_CARLA_PID','WMMPC_EXTERNAL_CARLA_PORT','WMMPC_EXTERNAL_CARLA_OWNER'):
            os.environ.pop(name,None)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--job-id',required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--carla-root',type=Path,required=True)
    p.add_argument('--phase',choices=('smoke','full'),required=True)
    p.add_argument('--gpu',type=int,choices=(0,1),default=0)
    p.add_argument('--worker-index',type=int,default=0);p.add_argument('--num-workers',type=int,default=8)
    p.add_argument('--slot',type=int,default=2250);p.add_argument('--seed',type=int,default=901)
    p.add_argument('--route-ids',default='1711,1773');p.add_argument('--routes-per-server',type=int,default=32)
    p.add_argument('--max-attempts',type=int,default=2)
    p.add_argument('--startup-timeout',type=float,default=600)
    p.add_argument('--no-progress-timeout',type=float,default=300)
    p.add_argument('--route-wall-timeout',type=float,default=5400)
    args=p.parse_args();check_job(args.job_id)
    if Path(sys.prefix).resolve()!=(ROOT/'.venvs/full-diffusion-carla').resolve():
        raise RuntimeError('Use the isolated CARLA environment')
    for name in ('checkpoint','output_root','carla_root'):
        value=getattr(args,name).resolve();value.relative_to(ROOT.resolve());setattr(args,name,value)
    if not (args.carla_root/'runtime_ready.json').is_file():raise RuntimeError('Private runtime is incomplete')
    if not 0<=args.worker_index<args.num_workers or not 1<=args.routes_per_server<=64:
        raise ValueError('Invalid worker or route batch size')
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().splitlines()
    if len(uuids)!=2:raise RuntimeError('Expected two allocation-visible GPUs')
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    if uuids[args.gpu] in occupied:raise RuntimeError('Target GPU is occupied')
    affinity=sorted(os.sched_getaffinity(0));half=len(affinity)//2
    if half<4:raise RuntimeError('Expected at least eight allocation CPUs')
    os.sched_setaffinity(0,affinity[args.gpu*half:(args.gpu+1)*half])
    elements=ET.parse(ROOT/'leaderboard/data/bench2drive220.xml').getroot().findall('route')
    available=[r.attrib['id'] for r in elements];towns={r.attrib['id']:r.attrib['town'] for r in elements}
    routes=args.route_ids.split(',') if args.phase=='smoke' else available[args.worker_index::args.num_workers]
    if len(routes)!=len(set(routes)) or not set(routes)<=set(available):raise ValueError('Invalid routes')
    if args.phase=='full':routes=sorted(routes,key=lambda route:towns[route])
    args.output_root.mkdir(parents=True,exist_ok=True)
    sources=['evaluation/run_full_diffusion_carla_warm.py','evaluation/run_full_diffusion_carla_fast.py',
             'evaluation/carla_fast_evaluator.py','evaluation/carla_lifecycle.py','evaluation/full_diffusion_fast_agent.py',
             'evaluation/fast_carla_io.py','evaluation/carla_static_roads.py',
             'method/full_diffusion/diffusion.py','method/full_diffusion/model.py',
             'method/full_diffusion/inference/policy.py','method/full_diffusion/inference/carla_agent.py']
    manifest={'job_id':args.job_id,'host':socket.gethostname(),'gpu_logical_index':args.gpu,'gpu_uuid':uuids[args.gpu],
              'affinity':sorted(os.sched_getaffinity(0)),'routes':routes,'seed':args.seed,
              'checkpoint':str(args.checkpoint),'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'inference_steps':100,'sampler':'Original VPFullDiffusion, unchanged','fresh_python_per_route':True,
              'fresh_agent_per_route':True,'full_world_reload_per_route':True,'routes_per_server':args.routes_per_server,
              'static_road_cache':False,
              'fresh_traffic_manager_per_route':True,
              'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
              'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'unix':time.time()}
    old=read(args.output_root/'manifest.json')
    if old and any(old.get(k)!=manifest[k] for k in ('routes','seed','checkpoint_sha256','source_sha256','routes_per_server')):
        raise RuntimeError('Resume identity differs')
    save(args.output_root/'manifest.json',manifest)
    def terminate(signum,frame):
        save(args.output_root/'controller_signal.json',{'signal':signum,'unix':time.time()})
        active=getattr(args,'_active_evaluator',None)
        if active is not None:stop_evaluator(active)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)
    locks=ROOT/'.runtime/carla_gpu_locks';locks.mkdir(parents=True,exist_ok=True)
    owner=None;sequence=0;used=0;finished=[];started=time.time();traffic_sequence=0
    try:
        with ExitStack() as stack:
            fcntl.flock(stack.enter_context((locks/(uuids[args.gpu]+'.lock')).open('a')),fcntl.LOCK_EX|fcntl.LOCK_NB)
            for port in range(2000+args.slot*10,2004+args.slot*10):
                fcntl.flock(stack.enter_context((locks/f'{socket.gethostname()}_{port}.lock').open('a')),fcntl.LOCK_EX|fcntl.LOCK_NB)
                with socket.socket() as probe:probe.bind(('127.0.0.1',port))
            for route in routes:
                directory=args.output_root/f'route_{route}';directory.mkdir(exist_ok=True)
                result=read(directory/'summary.json')
                if result and result['classification']=='driving_result':finished.append(result);continue
                for index in range(1,args.max_attempts+1):
                    attempt=directory/f'attempt_{index:02d}'
                    if attempt.exists():
                        previous=read(attempt/'summary.json')
                        if not previous:raise RuntimeError(f'Inspect incomplete attempt before resume: {attempt}')
                        if previous['classification']=='infrastructure_error':continue
                        raise RuntimeError(f'Inspect existing non-driving result before resume: {attempt}')
                    if owner is None or used>=args.routes_per_server:
                        if owner:owner.close();owner=None
                        cooperative_pause(args.output_root,args.job_id)
                        sequence+=1
                        while (args.output_root/f'servers/server_{sequence:03d}').exists():sequence+=1
                        owner=OwnedServer(args,uuids[args.gpu],args.output_root/f'servers/server_{sequence:03d}');used=0
                    args._external_server=owner.process
                    traffic_sequence+=1
                    args._traffic_manager_port=30000+args.worker_index*256+traffic_sequence
                    result=batch_trial(args,[route],attempt,uuids[args.gpu])[route];used+=1
                    if result['classification']=='infrastructure_error':
                        owner.close();owner=None
                        if index<args.max_attempts:continue
                    break
                if not result:raise RuntimeError('Infrastructure retries exhausted; inspect before resuming')
                save(directory/'summary.json',result);finished.append(result)
                save(args.output_root/'progress.json',{'done':len(finished),'total':len(routes),'unix':time.time(),
                     'driving_results':sum(x['classification']=='driving_result' for x in finished),
                     'errors':[x['route_id'] for x in finished if x['classification']!='driving_result']})
                if result['classification'] in ('agent_error','evaluation_error'):
                    save(args.output_root/'needs_attention.json',result);return 2
            final={'unix':time.time(),'elapsed_seconds':time.time()-started,'routes':len(finished),
                   'driving_results':sum(x['classification']=='driving_result' for x in finished),
                   'errors':[x['route_id'] for x in finished if x['classification']!='driving_result'],
                   'server_launches':sequence,'fresh_python_per_route':True}
            save(args.output_root/'completed.json',final)
            return int(bool(final['errors']))
    finally:
        if owner:owner.close()


if __name__=='__main__':raise SystemExit(main())
