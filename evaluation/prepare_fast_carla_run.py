"""Prepare a separate immutable 100-step run; does not launch or change LSF jobs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('evaluation/fast_carla_io.py', 'evaluation/carla_lifecycle.py', 'evaluation/carla_static_roads.py',
           'evaluation/full_diffusion_fast_agent.py', 'evaluation/carla_fast_evaluator.py',
           'evaluation/run_full_diffusion_carla_fast.py', 'evaluation/run_full_diffusion_carla_warm.py',
           'evaluation/launch_fast_carla_node.py',
           'evaluation/prepare_fast_carla_run.py', 'evaluation/verify_carla_lifecycle.py',
           'evaluation/validate_fast_at_route_boundary.py',
           'evaluation/run_full_diffusion_carla.py', 'evaluation/bench2drive.py',
           'evaluation/full_diffusion_diagnostic_agent.py',
           'evaluation/summarize_full_diffusion_carla.py',
           'method/full_diffusion/diffusion.py', 'method/full_diffusion/model.py',
           'method/full_diffusion/garage.py', 'method/full_diffusion/inference/policy.py',
           'method/full_diffusion/inference/carla_agent.py',
           'third_party/carla_garage/team_code/sensor_agent.py',
           'leaderboard/leaderboard/leaderboard_evaluator.py')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    args=parser.parse_args()
    checkpoint=args.checkpoint.resolve(); baseline=args.baseline_run.resolve(); run=args.run_root.resolve()
    checkpoint.relative_to(ROOT.resolve()); baseline.relative_to(ROOT.resolve()); run.relative_to(ROOT.resolve()/'results')
    if run.exists():raise FileExistsError(f'Use a new output directory: {run}')
    import torch
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state['config']['diffusion']['inference_steps']!=100:
        raise ValueError('The fixed experiment requires 100 denoising steps')
    epoch=state['progress']['epoch']
    topology=json.loads((baseline/'launch_topology.json').read_text())
    routes=[wroute for worker in topology['workers'] for wroute in worker['routes']]
    expected=[x.attrib['id'] for x in ET.parse(ROOT/'leaderboard/data/bench2drive220.xml').getroot().findall('route')]
    if len(routes)!=len(set(routes)) or set(routes)!=set(expected):raise ValueError('Invalid baseline route topology')
    if topology['num_workers']!=8 or len(topology['workers'])!=8:raise ValueError('Expected eight explicitly authorized GPU workers')
    for name in SOURCES:
        if not (ROOT/name).is_file():raise FileNotFoundError(name)
    run.mkdir(parents=True);(run/'checkpoint').mkdir();(run/'source_snapshot').mkdir()
    private=run/'checkpoint'/f'epoch_{epoch}.pt';shutil.copy2(checkpoint,private)
    digest=hashlib.sha256(private.read_bytes()).hexdigest()
    topology.update(checkpoint=str(private),checkpoint_sha256=digest,epoch=epoch,inference_steps=100,
                    created_utc=datetime.now(timezone.utc).isoformat(),profile='warm_carla_fresh_evaluator_original_sampler_100')
    for worker in topology['workers']:
        worker['slot']=2000+worker['worker_index']
    (run/'launch_topology.json').write_text(json.dumps(topology,indent=2)+'\n')
    hashes={}
    for name in SOURCES:
        src=ROOT/name;dst=run/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst);hashes[name]=hashlib.sha256(src.read_bytes()).hexdigest()
    (run/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    config={'checkpoint_sha256':digest,'epoch':epoch,'inference_steps':100,'simulation_hz':20,
            'route_count':220,'worker_count':8,'seed':topology['seed'],'default_routes_per_server':32,
            'precision':'Same autocast settings as the baseline; no compile/TF32 changes',
            'route_reset':'Full episode reload and fresh Python, agent, controller, temporal buffers and Traffic Manager per route',
            'sampler':'Original VPFullDiffusion class and implementation, unmodified',
            'static_roads':'Experimental disk graph cache disabled: exact native waypoint round-trip was not established',
            'server_reuse':'One worker-owned renderer, fresh evaluator process per route; bounded to 32 routes per server',
            'first_map_preload':'Disabled: tested command-line forms were ignored by this CARLA build',
            'logging':'Original sensor agent metric writes; passive lifecycle phase timings',
            'comparison':'Separate run directory; existing baseline remains unchanged'}
    (run/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
    (run/'checkpoint/integrity.json').write_text(json.dumps({'sha256':digest,'progress':state['progress'],
                                                          'inference_steps':100,'bytes':private.stat().st_size},indent=2)+'\n')
    for name in ('evaluation_requirements.lock','maps_and_versions.json'):
        if (baseline/name).exists():shutil.copy2(baseline/name,run/name)
    print(json.dumps({'prepared':str(run),'epoch':epoch,'inference_steps':100,'gpus':8,
                      'launch':'Inside each existing battach allocation: .venvs/full-diffusion-carla/bin/python evaluation/launch_fast_carla_node.py --job-id JOB --run-root '+str(run)},indent=2))


if __name__=='__main__':main()
