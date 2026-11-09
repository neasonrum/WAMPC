"""Checkpointed short measurement window, always followed by training resume."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .launch import ROOT,guard_context,write_json


def main():
    guard_context('13053006')
    run=Path('/fs/scratch/projects/common_ai/nix3sgh_cache/WMMPC/irwm_bev_t_dt05_h3_b32_train_20261106_v1')
    out=run/'inference_latency_20261106'
    assert (out/'real_inputs.pt').exists()
    status=json.loads((run/'node0.status.json').read_text())
    assert status['state']=='running' and not (run/'STOP').exists()
    marker='User-authorized isolated inference timing; resume automatically after measurement.'
    (run/'STOP').write_text(marker+'\n')
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        if json.loads((run/'node0.status.json').read_text())['state']=='completed' and (run/'stopped.json').exists():break
        time.sleep(2)
    else:raise RuntimeError('Training did not finish its graceful pause; inspect before touching anything')
    os.link(run/'latest.pt',out/'world_checkpoint.pt')
    write_json(out/'pause.json',json.loads((run/'stopped.json').read_text()))
    v3=Path('/fs/scratch/projects/common_ai/nix3sgh_cache/WMMPC/results/full_diffusionv3_dual_query64_automot_pid_train_20261104_v1/epoch_10.pt')
    command=[sys.executable,'-u','-m','method.irwm_bev_t.inference_latency','--phase','measure',
             '--output',str(out),'--world-checkpoint',str(out/'world_checkpoint.pt'),'--v3-checkpoint',str(v3)]
    started=datetime.now(timezone.utc).isoformat()
    rc=None
    try:
        with (out/'measurement.log').open('a') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        rc=result.returncode
    finally:
        assert (run/'STOP').read_text().strip()==marker
        os.replace(run/'STOP',out/'STOP.fulfilled')
        os.replace(run/'stopped.json',out/'stopped_before_measurement.json')
        resume=[sys.executable,'-u','-m','method.irwm_bev_t.launch','--job','13053006','--phase','train',
                '--config',str(ROOT/'method/irwm_bev_t/config.json'),'--output',str(run),'--tmux','--resume']
        restarted=subprocess.run(resume,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        write_json(out/'session.json',{'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),
                                     'measurement_return_code':rc,'resume_launcher_return_code':restarted.returncode,
                                     'resume_launcher_output':restarted.stdout})
        print(restarted.stdout,flush=True)
        if restarted.returncode:raise RuntimeError('Automatic resume launcher failed; inspect the recorded output')


if __name__=='__main__':main()
