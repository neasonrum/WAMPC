"""Guarded, durable launch inside the four user-supplied LSF allocations."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
AUTHORIZED={'13053006':'rng-dl01-w005','13047896':'rng-dl01-w002',
            '13047898':'rng-dl01-w007','13046830':'rng-dl01-w010'}


def guard_context(job):
    if job not in AUTHORIZED or socket.gethostname().split('.')[0]!=AUTHORIZED[job]:
        raise RuntimeError('Wrong authorized allocation host')
    if not re.search(rf'\b{job}\b',Path('/proc/self/cgroup').read_text()):
        raise RuntimeError('Not in the requested LSF job cgroup; enter with battach')


def write_json(path,value):
    path=Path(path)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--job',choices=AUTHORIZED,required=True)
    p.add_argument('--phase',choices=('acceptance','train'),required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tmux',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--port',type=int,default=29671)
    args=p.parse_args()
    guard_context(args.job)
    args.output=args.output.resolve()
    if not args.output.is_relative_to(ROOT):
        raise RuntimeError('Output must be inside WMMPC')
    args.output.mkdir(parents=True,exist_ok=True)
    node=list(AUTHORIZED).index(args.job)
    if args.tmux:
        log=args.output/f'node{node}.log'
        command=[sys.executable,'-u','-m','method.irwm_bev.launch','--job',args.job,
                 '--phase',args.phase,'--config',str(args.config.resolve()),'--output',str(args.output),'--port',str(args.port)]
        if args.resume:
            command.append('--resume')
        shell=f'cd {shlex.quote(str(ROOT))} && export PYTHONPATH={shlex.quote(str(ROOT))}; exec {shlex.join(command)} >> {shlex.quote(str(log))} 2>&1'
        session='irwm-'+args.phase+'-'+args.output.name
        subprocess.run(['tmux','-L','irwm'+args.job,'new-session','-d','-s',session,shell],check=True)
        print(json.dumps({'job':args.job,'node':node,'tmux':session,'log':str(log)}),flush=True)
        return
    with (args.output/f'.node{node}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import psutil
        probe="import torch,json; assert torch.cuda.device_count()==2; print(json.dumps([{'logical':i,'name':torch.cuda.get_device_name(i),'memory_gib':torch.cuda.get_device_properties(i).total_memory/2**30} for i in range(2)]))"
        # The probe exits before workers start; the durable controller holds no CUDA context.
        gpu_info=json.loads(subprocess.check_output([sys.executable,'-c',probe],text=True))
        cg=Path('/proc/self/cgroup').read_text().strip().split(':')[-1]
        parent=Path('/sys/fs/cgroup')/cg.lstrip('/')
        limits={}
        for pth in [parent,parent.parent]:
            for name in ('memory.max','memory.current'):
                if (pth/name).exists():
                    limits[str(pth/name)]=(pth/name).read_text().strip()
        address=socket.gethostbyname(socket.gethostname())
        names=[n for n,aa in psutil.net_if_addrs().items() if any(a.family==socket.AF_INET and a.address==address for a in aa)]
        if len(names)!=1:
            raise RuntimeError('Cannot determine cluster interface')
        env=os.environ.copy()
        env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
                   PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                   NCCL_SOCKET_IFNAME=names[0],GLOO_SOCKET_IFNAME=names[0],NCCL_IB_DISABLE='1',
                   NCCL_NVLS_ENABLE='0',TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
        # Cgroup isolation already maps to logical 0,1. Never set host physical indices.
        command=[sys.executable,'-u','-m','torch.distributed.run','--nnodes=4','--nproc_per_node=2',
                 f'--node_rank={node}','--master_addr=10.124.70.47',f'--master_port={args.port}',
                 '--max_restarts=0','-m','method.irwm_bev.train','--config',str(args.config.resolve()),
                 '--output',str(args.output),'--phase',args.phase]
        if args.resume:
            command.append('--resume')
        status={'state':'running','job':args.job,'host':socket.gethostname(),'node':node,'pid':os.getpid(),
                'gpus':gpu_info,'cpu_affinity':sorted(os.sched_getaffinity(0)),'memory':limits,
                'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'network':names[0],
                'started_utc':datetime.now(timezone.utc).isoformat(),'command':command}
        write_json(args.output/f'node{node}.status.json',status)
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL)
        status['torchrun_pid']=child.pid
        write_json(args.output/f'node{node}.status.json',status)
        rc=child.wait()
        status.update(return_code=rc,state='completed' if rc==0 else 'failed',finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output/f'node{node}.status.json',status)
        sys.exit(rc)


if __name__=='__main__':
    main()
