"""CPU correctness checks for loading adapters and route isolation."""
import copy
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
for relative in ('', 'leaderboard', 'scenario_runner'):
    sys.path.insert(0, str(ROOT / relative))
os.environ.setdefault('OMP_NUM_THREADS', '1')
from evaluation.carla_lifecycle import Lifecycle, instrument_evaluator, map_asset, ordered_route_file
from evaluation.fast_carla_io import replace_startup_sleep
from evaluation.run_full_diffusion_carla_fast import publish_completed

OUTPUT = ROOT / 'results/full_diffusion_carla_20261031_lifecycle_validation'


class LifecycleChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=OUTPUT)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    @staticmethod
    def world(town='Town12', actors=()):
        return types.SimpleNamespace(id=42, get_map=lambda: types.SimpleNamespace(name='/Game/Carla/Maps/'+town),
                                     get_actors=lambda: actors,
                                     get_snapshot=lambda: types.SimpleNamespace(frame=5))

    def test_preload_is_only_used_for_unused_first_episode(self):
        current = self.world()
        loads = []
        def load(town, reset_settings):
            loads.append((town, reset_settings))
            return self.world(town)
        evaluator = types.SimpleNamespace(client=types.SimpleNamespace(get_world=lambda: current, load_world=load))
        lifecycle = Lifecycle(self.folder, 'Town12')
        lifecycle.routes_started = 1
        self.assertIs(lifecycle.load_world(evaluator, None, 'Town12'), current)
        self.assertEqual(loads, [])
        lifecycle.routes_started = 2
        lifecycle.load_world(evaluator, None, 'Town12')
        lifecycle.routes_started = 3
        lifecycle.load_world(evaluator, None, 'Town13')
        self.assertEqual(loads, [('Town12', False), ('Town13', False)])
        events = [json.loads(line) for line in (self.folder/'lifecycle_events.jsonl').read_text().splitlines()]
        self.assertEqual([x['mode'] for x in events if x['event']=='world_load_finished'],
                         ['fresh_preloaded_first_episode', 'full_episode_reload', 'full_episode_reload'])

    def test_unexpected_map_or_route_actor_forces_reload(self):
        for town, actors in [('Town13', []), ('Town12', [types.SimpleNamespace(id=1,type_id='sensor.camera.rgb')])]:
            calls = []
            evaluator = types.SimpleNamespace(client=types.SimpleNamespace(
                get_world=lambda:self.world(town, actors),
                load_world=lambda requested,reset_settings:(calls.append(requested) or self.world(requested))))
            lifecycle = Lifecycle(self.folder, 'Town12'); lifecycle.routes_started=1
            lifecycle.load_world(evaluator,None,'Town12')
            self.assertEqual(calls,['Town12'])

    def test_route_definitions_survive_map_grouping(self):
        original=ROOT/'leaderboard/data/bench2drive220.xml'
        routes={r.attrib['id']:r for r in ET.parse(original).getroot().findall('route')}
        selected=list(routes)[::31][::-1]
        destination=self.folder/'routes.xml'
        ordered_route_file(original,selected,destination)
        after=ET.parse(destination).getroot().findall('route')
        self.assertEqual([r.attrib['id'] for r in after],selected)
        for route in after:
            self.assertEqual(ET.tostring(route),ET.tostring(routes[route.attrib['id']]))
        with self.assertRaises(ValueError):ordered_route_file(original,['1711','1711'],destination)
        self.assertEqual(map_asset('Town12'),'/Game/Carla/Maps/Town12/Town12')
        self.assertEqual(map_asset('Town01'),'/Game/Carla/Maps/Town01')
        with self.assertRaises(ValueError):map_asset('Town12_Opt')

    def test_pinned_source_changes_only_lifecycle_sites(self):
        source=(ROOT/'leaderboard/leaderboard/leaderboard_evaluator.py').read_text()
        changed=replace_startup_sleep(instrument_evaluator(source))
        restored=changed.replace('        self.world = _wmmpc_load_world(self, args, town)',
                                '        self.world = self.client.load_world(town, reset_settings=False)')
        restored=restored.replace('        _wmmpc_wait_for_server(self.server, args.host, args.port)',
                                  '        time.sleep(30)')
        self.assertEqual(source,restored)
        with self.assertRaises(RuntimeError):instrument_evaluator(source+'\n')

    def test_process_globals_are_reset_between_routes(self):
        import numpy as np
        import torch
        import py_trees
        import leaderboard.utils.parked_vehicles as parked
        parking_before={name:copy.deepcopy(value) for name,value in vars(parked).items()
                        if name.startswith('Town') and isinstance(value,list)}
        board=vars(py_trees.blackboard.Blackboard())
        board_before=copy.deepcopy(board)
        observations=[]
        key=next(name for name,value in parking_before.items() if value)
        def original(evaluator,args,config):
            observations.append((random.random(),np.random.random(),float(torch.rand(())),
                                 len(getattr(parked,key)),copy.deepcopy(board)))
            getattr(parked,key).clear()
            board['previous_route_should_not_leak']=True
            return False
        class Evaluator:pass
        Evaluator._load_and_run_scenario=original
        lifecycle=Lifecycle(self.folder,'Town12')
        namespace={'LeaderboardEvaluator':Evaluator}
        try:
            with patch('evaluation.carla_static_roads.install_static_road_cache'):
                lifecycle.install(namespace)
            evaluator=Evaluator();evaluator.world=self.world()
            for name in ['RouteScenario_1711','RouteScenario_1773']:
                evaluator._load_and_run_scenario(types.SimpleNamespace(agent_config='model.pt+old'),
                                                types.SimpleNamespace(name=name,town='Town12'))
            self.assertEqual(observations[0],observations[1])
            self.assertEqual(observations[0][3],len(parking_before[key]))
        finally:
            for name,value in parking_before.items():setattr(parked,name,value)
            board.clear();board.update(board_before)

    def test_static_road_cache_keeps_fresh_decisions_and_invalidates_changed_geometry(self):
        from evaluation.carla_static_roads import StaticRoadGraphCache
        builds=[]
        def constructor(world_map,resolution):
            builds.append((world_map.name,resolution))
            return types.SimpleNamespace(_graph={'static':'topology'},_previous_decision='VOID',_intersection_end_node=-1)
        cache=StaticRoadGraphCache(constructor,lambda *args,**kwargs:None)
        world_map=types.SimpleNamespace(name='Town12',to_opendrive=lambda:'same immutable geometry')
        first=cache(world_map,1.0);first._previous_decision='LEFT';first._intersection_end_node=17
        second=cache(world_map,1.0)
        self.assertIs(first._graph,second._graph)
        self.assertEqual((second._previous_decision,second._intersection_end_node),('VOID',-1))
        self.assertEqual(len(builds),1)
        cache(world_map,2.0);self.assertEqual(len(builds),2)
        world_map.to_opendrive=lambda:'changed geometry'
        cache(world_map,1.0);self.assertEqual(len(builds),3)
        self.assertEqual(len(cache.templates),1)

    def test_earlier_score_survives_later_server_failure(self):
        attempt=self.folder/'attempt_01'
        metric=attempt/'agent_data/completed/metric_info.json'
        metric.parent.mkdir(parents=True);metric.write_text('{}')
        record={'route_id':'RouteScenario_1711_rep0','status':'Completed',
                'save_name':'completed','meta':{'duration_game':10},'scores':{}}
        result={'entry_status':'Rejected','_checkpoint':{'records':[record]}}
        published,_=publish_completed(result,['1711','1773'],attempt,self.folder,0)
        self.assertEqual(published,['1711'])
        self.assertEqual(json.loads((self.folder/'route_1711/summary.json').read_text())['classification'], 'driving_result')


if __name__=='__main__':
    OUTPUT.mkdir(parents=True,exist_ok=True)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LifecycleChecks))
    (OUTPUT/'cpu_checks.json').write_text(json.dumps({'tests':result.testsRun,'success':result.wasSuccessful(),
        'errors':len(result.errors),'failures':len(result.failures), 'sampler':'Original VPFullDiffusion, 100 steps'},indent=2)+'\n')
    raise SystemExit(int(not result.wasSuccessful()))
