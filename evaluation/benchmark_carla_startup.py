"""Measure CARLA loading only, on an idle logical GPU in the current allocation."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.bench2drive import check_job
from evaluation.fast_carla_io import wait_for_server
from evaluation.carla_static_roads import StaticRoadGraphCache


def verify_roads(world,cache,configs,constructor):
    reports=[]
    for resolution in (1.0,2.0):
        started=time.monotonic();fresh=constructor(world.get_map(),resolution)
        fresh_seconds=time.monotonic()-started
        def signature(planner,config):
            result=[]
            for left,right in zip(config.keypoints,config.keypoints[1:]):
                for waypoint,option in planner.trace_route(left,right):
                    transform=waypoint.transform;loc=transform.location;rot=transform.rotation
                    result.append((loc.x,loc.y,loc.z,rot.pitch,rot.yaw,rot.roll,option.value))
            return result
        for config in configs:
            started=time.monotonic();reused=cache(world.get_map(),resolution)
            cache_seconds=time.monotonic()-started
            a=signature(copy.copy(fresh),config);b=signature(reused,config)
            if a!=b:raise RuntimeError('Cached static map changed route geometry: '+config.name)
            reports.append({'route':config.name,'resolution':resolution,'points':len(a),
                            'all_coordinates_and_options_exactly_equal':True,
                            'fresh_graph_seconds':fresh_seconds,'cached_graph_seconds':cache_seconds,
                            'world_id':world.id})
    return reports


def main():
    p=argparse.ArgumentParser();p.add_argument('--job-id',required=True)
    p.add_argument('--gpu',type=int,choices=(0,1),default=1)
    p.add_argument('--output-root',type=Path,required=True)
    args=p.parse_args();check_job(args.job_id)
    folder=args.output_root.resolve();folder.relative_to(ROOT/'results')
    if folder.exists():raise FileExistsError(folder)
    runtime=ROOT/'.runtime/carla_0.9.15_v1'
    if not (runtime/'runtime_ready.json').is_file():raise RuntimeError('Private runtime incomplete')
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    if len(rows)!=2:raise RuntimeError('Expected two allocation GPUs')
    uuid,memory=rows[args.gpu].split(',')
    if int(memory.strip())>100:raise RuntimeError('Target GPU still occupied: '+rows[args.gpu])
    affinity=sorted(os.sched_getaffinity(0));half=len(affinity)//2
    if half<4:raise RuntimeError('Unexpected CPU affinity')
    os.sched_setaffinity(0,affinity[args.gpu*half:(args.gpu+1)*half])
    folder.mkdir(parents=True)
    os.environ['CUDA_VISIBLE_DEVICES']=uuid.strip()
    os.environ['VK_ICD_FILENAMES']='/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json'
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['TMPDIR']=str(folder/'tmp');(folder/'tmp').mkdir()
    sys.path.insert(0,str(runtime/'PythonAPI/carla'))
    sys.path.insert(0,str(ROOT/'leaderboard'));sys.path.insert(0,str(ROOT/'scenario_runner'))
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from leaderboard.utils.route_parser import RouteParser
    configs=RouteParser.parse_routes_file(str(ROOT/'leaderboard/data/bench2drive220.xml'),'1711,1773')
    binary=runtime/'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping'
    records=[]
    variants=[('baseline_loading',['CarlaUE4']),('repeat_cached_loading',['CarlaUE4'])]
    for index,(variant,positionals) in enumerate(variants):
        log=folder/variant;log.mkdir()
        for name in ['home','cache','unreal_user']:(log/name).mkdir()
        port=24500+args.gpu*20+index*4
        command=[str(binary),*positionals,'-RenderOffScreen','-nosound',f'-carla-rpc-port={port}',
                 f'-graphicsadapter={args.gpu}',f'-userdir={log/"unreal_user"}',
                 f'-abslog={log/"CarlaUE4.log"}','-unattended','-nosplash']
        env=os.environ.copy();env['HOME']=str(log/'home');env['XDG_CACHE_HOME']=str(log/'cache')
        started=time.monotonic()
        with (log/'stdout.log').open('wb') as output:
            server=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            record={'variant':variant,'command':command,'pid':server.pid,'gpu_uuid':uuid.strip()}
            try:
                ready=wait_for_server(server,'localhost',port,log,timeout=180)
                client=carla.Client('localhost',port);client.set_timeout(300)
                world=client.get_world();town=world.get_map().name.split('/')[-1]
                record.update(ready=ready,initial_town=town,
                              startup_seconds=time.monotonic()-started,loads=[])
                print(json.dumps(record),flush=True)
                settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.05
                settings.deterministic_ragdolls=True;settings.spectator_as_ego=False
                world.apply_settings(settings)
                if town!='Town12':
                    before=time.monotonic();world=client.load_world('Town12',reset_settings=False)
                    record['loads'].append({'target':'Town12','previous':town,
                                            'phase':'first_required_map',
                                            'elapsed_seconds':time.monotonic()-before})
                cache_events=[]
                callback=lambda event,**data:cache_events.append({'event':event,**data})
                cache=StaticRoadGraphCache(GlobalRoutePlanner,callback,ROOT/'.runtime/static_road_graphs_v1')
                record['static_road_checks']=verify_roads(world,cache,configs,GlobalRoutePlanner)
                for target in ['Town12','Town13']:
                    before=time.monotonic();loaded=client.load_world(target,reset_settings=False)
                    actual=loaded.get_map().name.split('/')[-1]
                    record['loads'].append({'target':target,'actual':actual,'elapsed_seconds':time.monotonic()-before,
                                            'world_id':loaded.id,'frame':loaded.get_snapshot().frame})
                    print(json.dumps(record['loads'][-1]),flush=True)
                    if target=='Town12':
                        # A new cache object forces a disk read against the
                        # new episode, as in a separate evaluator process.
                        cache=StaticRoadGraphCache(GlobalRoutePlanner,callback,ROOT/'.runtime/static_road_graphs_v1')
                        record['static_road_checks'].extend(verify_roads(loaded,cache,configs,GlobalRoutePlanner))
                record['static_road_events']=cache_events
            except Exception as exc:
                record['error']=repr(exc)
                print(json.dumps(record),flush=True)
            finally:
                if server.poll() is None:
                    os.killpg(server.pid,signal.SIGTERM)
                    try:server.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=5)
                record['returncode']=server.returncode
                records.append(record)
                (folder/'loading_measurements.json').write_text(json.dumps(records,indent=2)+'\n')
    return int(not all(len(r.get('static_road_checks',[]))==8 and len(r.get('loads',[]))==3
                       and not r.get('error') for r in records))


if __name__=='__main__':raise SystemExit(main())
