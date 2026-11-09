"""Borrow only a cooperatively paused fast worker; never use process signals."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.bench2drive import check_job
from evaluation.run_full_diffusion_carla import read,save


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--job-id',required=True)
    p.add_argument('--baseline-run',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--worker-index',type=int,default=0)
    p.add_argument('--max-wait',type=float,default=1200)
    args=p.parse_args();check_job(args.job_id)
    run=args.baseline_run.resolve();output=args.output_root.resolve()
    run.relative_to(ROOT.resolve()/'results');output.relative_to(ROOT.resolve()/'results')
    worker=run/f'full/worker{args.worker_index}'
    manifest=read(worker/'manifest.json',{})
    if not manifest.get('cooperative_pause_supported'):
        raise RuntimeError('This worker has no cooperative boundary gate. Refusing to borrow a live baseline GPU. Wait for a fully idle GPU instead.')
    if manifest['job_id']!=args.job_id:raise RuntimeError('Wrong allocation')
    output.mkdir(parents=True,exist_ok=True)
    token=uuid.uuid4().hex
    request=worker/'pause_request.json';temporary=worker/f'.pause_request.{token}.tmp'
    temporary.write_text(json.dumps({'job_id':args.job_id,'token':token,'unix':time.time()}))
    try:
        os.link(temporary,request)
    finally:temporary.unlink()
    receipt={'job_id':args.job_id,'worker':args.worker_index,'token':token,
             'started_unix':time.time(),'coordination':'cooperative boundary acknowledgment',
             'signals_sent':False,'allocation_cancelled':False}
    def terminate(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)
    try:
        deadline=time.monotonic()+args.max_wait
        while time.monotonic()<deadline:
            acknowledgment=read(worker/'pause_ack.json',{})
            if acknowledgment.get('token')==token and acknowledgment.get('state')=='paused':break
            time.sleep(.2)
        else:raise TimeoutError('The current batch has not naturally reached its boundary')
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if manifest['gpu_uuid'] in apps:raise RuntimeError('The acknowledged GPU is still occupied; refusing to run validation')
        receipt['acknowledged_unix']=time.time();save(output/'cooperative_validation.json',receipt)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=manifest['gpu_uuid'],OMP_NUM_THREADS='1',
                                        MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (output/'gpu_checks.log').open('w') as log:
            result=subprocess.run([sys.executable,'-u',str(ROOT/'evaluation/verify_fast_overheads.py'),
                                   '--device','cuda:0'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        receipt.update(gpu_checks_returncode=result.returncode,completed_unix=time.time())
        return result.returncode
    finally:
        if read(request,{}).get('token')==token:request.unlink()
        receipt.update(request_released=True,released_unix=time.time())
        save(output/'cooperative_validation.json',receipt)


if __name__=='__main__':raise SystemExit(main())
