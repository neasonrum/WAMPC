"""Paired closed-loop loading test inside an idle, existing two-GPU allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.bench2drive import check_job
from evaluation.run_full_diffusion_carla import trial,save


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--job-id',required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--mode',choices=('launch','baseline','optimized'),default='launch')
    args=p.parse_args();check_job(args.job_id)
    folder=args.output_root.resolve();folder.relative_to(ROOT/'results')
    runtime=ROOT/'.runtime/carla_0.9.15_v1'
    if not (runtime/'runtime_ready.json').is_file():raise RuntimeError('Private runtime is incomplete')
    checkpoint=ROOT/'results/full_diffusion_carla_20261031_epoch10/checkpoint/epoch_10.pt'
    affinity=sorted(os.sched_getaffinity(0))
    if len(affinity)<8:raise RuntimeError('Expected at least eight allocation CPUs')
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().splitlines()
    if len(uuids)!=2:raise RuntimeError('Expected two allocation-visible GPUs')
    routes=['1711','1773']
    if args.mode=='launch':
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True).strip()
        if apps:raise RuntimeError('Allocation GPUs are occupied: '+apps)
        if folder.exists():raise FileExistsError(folder)
        folder.mkdir(parents=True)
        save(folder/'comparison_config.json',{'job_id':args.job_id,'host':socket.gethostname(),
             'routes':routes,'affinity':affinity,'gpus':uuids,'inference_steps':100,
             'sampler':'Original VPFullDiffusion, unchanged','checkpoint':str(checkpoint),
             'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
             'runtime':str(runtime),'expected_peak_host_memory_gib':32,'unix':time.time(),
             'baseline':'Fresh server and full reload per route, original metric logging',
             'optimized':'One server, first map preload, full reload for next route, original metric logging'})
        for mode in ('baseline','optimized'):
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--job-id',args.job_id,
                     '--output-root',str(folder),'--mode',mode]
            shell='exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '+shlex.join(command)
            shell+=' > '+shlex.quote(str(folder/(mode+'.controller.log')))+' 2>&1'
            subprocess.run(['tmux','-S',str(ROOT/f'.cl_{args.job_id}.sock'),'new-session','-d',
                            '-s','loading_'+mode,shell],cwd=ROOT,check=True)
        print(json.dumps({'launched':str(folder),'job_id':args.job_id,'routes':routes}),flush=True)
        return 0
    os.environ['CARLA_ROOT']=str(runtime)
    if args.mode=='optimized':
        command=[sys.executable,'-u',str(ROOT/'evaluation/run_full_diffusion_carla_fast.py'),
                 '--job-id',args.job_id,'--phase','smoke','--gpu','1','--worker-index','1','--num-workers','8',
                 '--slot','2201','--seed','901','--route-ids',','.join(routes),'--routes-per-server','32',
                 '--carla-root',str(runtime),'--checkpoint',str(checkpoint),'--output-root',str(folder/'optimized')]
        os.execv(sys.executable,command)
    os.sched_setaffinity(0,affinity[:len(affinity)//2])
    output=folder/'baseline';output.mkdir()
    settings=SimpleNamespace(job_id=args.job_id,checkpoint=checkpoint,output_root=output,
                             seed=901,slot=2200,gpu=0,phase='smoke',worker_index=0,
                             startup_timeout=600,no_progress_timeout=300,route_wall_timeout=5400)
    started=time.time();summaries=[]
    for route in routes:
        attempt=output/f'route_{route}/attempt_01'
        result=trial(settings,route,attempt,uuids[0])
        save(attempt.parent/'summary.json',result);summaries.append(result)
        if result['classification']!='driving_result':break
    save(output/'completed.json',{'elapsed_seconds':time.time()-started,'routes':len(summaries),
                                'driving_results':sum(s['classification']=='driving_result' for s in summaries),
                                'errors':[s['route_id'] for s in summaries if s['classification']!='driving_result']})
    return int(len(summaries)!=len(routes) or any(s['classification']!='driving_result' for s in summaries))


if __name__=='__main__':raise SystemExit(main())
