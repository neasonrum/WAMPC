"""Validate road caches against fresh CARLA maps, without starting a renderer."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
for relative in ('','leaderboard','scenario_runner','.runtime/carla_0.9.15_v1/PythonAPI/carla'):
    sys.path.insert(0,str(ROOT/relative))
from evaluation.bench2drive import check_job
from evaluation.carla_static_roads import StaticRoadGraphCache


def signature(planner,config):
    result=[]
    for left,right in zip(config.keypoints,config.keypoints[1:]):
        for waypoint,option in planner.trace_route(left,right):
            transform=waypoint.transform;loc=transform.location;rot=transform.rotation
            result.append((loc.x,loc.y,loc.z,rot.pitch,rot.yaw,rot.roll,option.value))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--job-id',required=True);args=p.parse_args();check_job(args.job_id)
    affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,affinity[len(affinity)//2:])
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from leaderboard.utils.route_parser import RouteParser
    text=(ROOT/'.runtime/carla_0.9.15_v1/CarlaUE4/Content/Carla/Maps/Town12/OpenDrive/Town12.xodr').read_bytes().decode('utf-8')
    maps=[carla.Map('Carla/Maps/Town12/Town12',text),carla.Map('Carla/Maps/Town12/Town12',text)]
    configs=[c for c in RouteParser.parse_routes_file(str(ROOT/'leaderboard/data/bench2drive220.xml')) if c.town=='Town12']
    events=[]
    def event(name,**data):
        r={'event':name,**data};events.append(r);print(json.dumps(r),flush=True)
    report={'job_id':args.job_id,'town':'Town12','map_sha256':hashlib.sha256(maps[0].to_opendrive().encode()).hexdigest(),
            'routes':len(configs),'comparisons':[]}
    output=ROOT/'results/full_diffusion_carla_20261031_lifecycle_validation/disk_road_checks.json'
    for index,world_map in enumerate(maps):
        cached=StaticRoadGraphCache(GlobalRoutePlanner,event,ROOT/'.runtime/static_road_graphs_v1')
        for resolution in (1.0,2.0):
            started=time.monotonic();fresh=GlobalRoutePlanner(world_map,resolution);fresh_seconds=time.monotonic()-started
            started=time.monotonic();prototype=cached(world_map,resolution);cached_seconds=time.monotonic()-started
            for config in configs:
                a=signature(copy.copy(fresh),config);b=signature(copy.copy(prototype),config)
                if a!=b:raise RuntimeError('Road geometry changed: '+config.name)
                report['comparisons'].append({'map_instance':index,'route':config.name,'resolution':resolution,
                                             'points':len(a),'all_coordinates_and_options_exactly_equal':True})
            print(json.dumps({'map_instance':index,'resolution':resolution,'fresh_seconds':fresh_seconds,
                              'cached_seconds':cached_seconds,'routes_compared':len(configs)}),flush=True)
            report.setdefault('timings',[]).append({'map_instance':index,'resolution':resolution,
                                                    'fresh_seconds':fresh_seconds,'cached_seconds':cached_seconds})
            report['events']=events;output.write_text(json.dumps(report,indent=2)+'\n')
    hits=[e for e in events if e['event']=='static_road_disk_cache' and e['cache_hit']]
    report['success']=len(hits)>=2
    output.write_text(json.dumps(report,indent=2)+'\n')
    return int(not report['success'])


if __name__=='__main__':raise SystemExit(main())
