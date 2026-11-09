"""Documented recovery of the two known shared-export failures, no job changes.

One extra attempt is allowed only for the exact two infrastructure failures
below. Original attempts and driving outcomes are retained without alteration.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.bench2drive import check_job
from evaluation.run_full_diffusion_carla import read,save,trial


def main():
    p=argparse.ArgumentParser();p.add_argument('--job-id',required=True)
    p.add_argument('--worker',type=int,choices=(6,7))
    p.add_argument('--launch-workers',default='6,7')
    args=p.parse_args();check_job(args.job_id)
    if args.job_id!='13047896' or socket.gethostname().split('.')[0]!='rng-dl01-w002':
        raise RuntimeError('This recovery is specific to the inspected w002 export incident')
    runtime=ROOT/'.runtime/carla_0.9.15_v1'
    integrity=read(runtime/'runtime_ready.json')
    if not integrity:raise RuntimeError('Private runtime has no integrity manifest')
    base=ROOT/'results/full_diffusion_carla_20261031_epoch10'
    checkpoint=base/'checkpoint/epoch_10.pt'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!='031fd8b5a120200de81bf14a8f170d191da3f2b3ed32691fded2a46120c8cac3':
        raise RuntimeError('Checkpoint changed')
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().splitlines()
    expected=['GPU-178dbb9f-517a-16af-9122-031d88c03b5e','GPU-f653bbe0-9522-f398-aff0-3931b9958f47']
    if uuids!=expected:raise RuntimeError('Allocation-visible GPU mapping changed')
    if args.worker is None:
        selected=[int(x) for x in args.launch_workers.split(',')]
        if len(selected)!=len(set(selected)) or not set(selected)<={6,7}:
            raise ValueError('Recovery launch workers must be 6 and/or 7')
        occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True).strip()
        if any(uuids[worker-6] in occupied for worker in selected):
            raise RuntimeError('Selected recovery GPU is occupied: '+occupied)
        for worker in selected:
            output=base/f'full/worker{worker}'
            recovery=output/'recovery_storage_export_20261031'
            recovery.mkdir(exist_ok=False)
            for name in ('manifest.json','progress.json','status.json'):
                if (output/name).exists():shutil.copy2(output/name,recovery/name)
            old_log=base/f'worker{worker}.controller.log'
            shutil.copy2(old_log,recovery/'controller_before_recovery.log')
            incident=read(output/f'route_{2273 if worker==6 else 2709}/summary.json')
            if not incident or incident['classification']!='infrastructure_error':
                raise RuntimeError('Expected the inspected infrastructure failure')
            shutil.copy2(output/f'route_{incident["route_id"]}/summary.json',recovery/'failed_route_summary.json')
            preflight_route='2509' if worker==6 else '2913'
            attempt=output/f'route_{preflight_route}/attempt_01'
            # The failed build_plan created only these empty directories. There
            # was no evaluator, server, sensor or driving result in this attempt.
            if set(x.name for x in attempt.iterdir())!={'tmp','cache','matplotlib'} or any(any(x.iterdir()) for x in attempt.iterdir()):
                raise RuntimeError('Preflight attempt has unexpected evidence; inspect manually')
            save(attempt/'summary.json',{'route_id':preflight_route,'classification':'infrastructure_error',
                 'attempt':str(attempt),'record':None,'status':'Not started',
                 'supervisor_reason':'runtime_preflight_stale_file_handle',
                 'driving_started':False,'recovery_evidence':str(recovery/'controller_before_recovery.log')})
            save(recovery/'recovery_plan.json',{'job_id':args.job_id,'worker':worker,'unix':time.time(),
                 'cause':'Shared CARLA dataset export returned errno 116 on w002; old controller exited in build_plan',
                 'runtime':str(runtime),'runtime_integrity':integrity,
                 'extra_attempt_route':incident['route_id'],'extra_attempt_index':3,
                 'policy':'One documented extra infrastructure attempt for this incident; no driving failure reruns',
                 'checkpoint_sha256':'031fd8b5a120200de81bf14a8f170d191da3f2b3ed32691fded2a46120c8cac3'})
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--job-id',args.job_id,'--worker',str(worker)]
            shell='exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '+shlex.join(command)
            shell+=' > '+shlex.quote(str(recovery/'recovery.controller.log'))+' 2>&1'
            subprocess.run(['tmux','-S',str(ROOT/f'.e10_{args.job_id}.sock'),'new-session','-d',
                            '-s',f'e10_worker_{worker}',shell],cwd=ROOT,check=True)
        print('Recovered workers launched inside the existing allocation',flush=True)
        return 0
    worker=args.worker;gpu=worker-6;output=base/f'full/worker{worker}'
    recovery=output/'recovery_storage_export_20261031'
    route='2273' if worker==6 else '2709'
    os.environ['CARLA_ROOT']=str(runtime)
    affinity=sorted(os.sched_getaffinity(0))
    if len(affinity)<8:raise RuntimeError('Unexpected allocation CPU affinity')
    half=len(affinity)//2
    os.sched_setaffinity(0,affinity[gpu*half:(gpu+1)*half])
    settings=SimpleNamespace(job_id=args.job_id,checkpoint=checkpoint,output_root=output,seed=901,
         slot=1900+worker,gpu=gpu,phase='full',worker_index=worker,
         startup_timeout=600,no_progress_timeout=300,route_wall_timeout=5400)
    result=trial(settings,route,output/f'route_{route}/attempt_03',uuids[gpu])
    save(recovery/'extra_attempt_result.json',result)
    if result['classification']!='driving_result':
        save(output/'needs_attention.json',result)
        return 2
    save(output/f'route_{route}/summary.json',result)
    save(recovery/'recovery_completed.json',{'unix':time.time(),'driving_result':True,'route':route})
    # The ordinary controller expects the full allocation affinity and splits
    # it itself, so restore that mask before exec. Its sources stay unchanged.
    os.sched_setaffinity(0,affinity)
    command=[sys.executable,'-u',str(ROOT/'evaluation/run_full_diffusion_carla.py'),
             '--job-id',args.job_id,'--phase','full','--gpu',str(gpu),'--worker-index',str(worker),
             '--num-workers','8','--slot',str(1900+worker),'--seed','901',
             '--checkpoint',str(checkpoint),'--output-root',str(output),'--max-attempts','2']
    os.execv(sys.executable,command)


if __name__=='__main__':raise SystemExit(main())
