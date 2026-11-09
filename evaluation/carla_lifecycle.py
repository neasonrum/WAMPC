"""CARLA process/map loading adapters. Planner and scoring are never patched."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time
import xml.etree.ElementTree as ET

EVALUATOR_SHA256 = '0c179936815471bef2c250fd13ab57f7bc6bd866d00f6d7e69d60db6ec6d0887'


def ordered_route_file(original, route_ids, destination):
    """Retain complete route definitions while choosing an explicit batch order."""
    tree = ET.parse(original)
    root = tree.getroot()
    by_id = {element.attrib['id']: element for element in root.findall('route')}
    if len(route_ids) != len(set(route_ids)) or not set(route_ids) <= set(by_id):
        raise ValueError('Unknown or duplicate route IDs')
    for element in list(root.findall('route')):
        root.remove(element)
    for route_id in route_ids:
        root.append(by_id[route_id])
    tree.write(destination, encoding='utf-8', xml_declaration=True)


def first_town(arguments):
    routes = Path(arguments[arguments.index('--routes') + 1])
    subset = arguments[arguments.index('--routes-subset') + 1].split(',') if '--routes-subset' in arguments else None
    for route in ET.parse(routes).getroot().findall('route'):
        if subset is None or route.attrib['id'] in subset:
            return route.attrib['town']
    raise ValueError('No matching route to preload')


def map_asset(town):
    # These are the normal 0.9.15 assets, including the directory in large towns.
    if town in ('Town11', 'Town12', 'Town13', 'Town15'):
        return f'/Game/Carla/Maps/{town}/{town}'
    if town not in ('Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD'):
        raise ValueError(f'Unverified CARLA map: {town}')
    return f'/Game/Carla/Maps/{town}'


def instrument_evaluator(source):
    """Intercept only the pinned evaluator's one world-load expression."""
    if hashlib.sha256(source.encode()).hexdigest() != EVALUATOR_SHA256:
        raise RuntimeError('Upstream evaluator changed; review the lifecycle adapter')
    target = '        self.world = self.client.load_world(town, reset_settings=False)'
    if source.count(target) != 1:
        raise RuntimeError('Expected one upstream map-load expression')
    changed = source.replace(target, '        self.world = _wmmpc_load_world(self, args, town)', 1)
    ast.parse(changed)
    return changed


class Lifecycle:
    """One server per batch; only its unused, preloaded first episode is reused.

    Later routes still reload the episode, including when the town is unchanged.
    That reset clears CARLA's physics, actor, sensor, weather and simulation RNG
    state; skipping it is not established as equivalent to the benchmark.
    """
    def __init__(self, directory, initial_town, use_initial_world=True):
        self.directory = Path(directory)
        self.initial_town = initial_town
        self.use_initial_world = use_initial_world
        self.routes_started = 0
        self.current_route = None
        self._python_state = None
        self._numpy_state = None
        self._torch_state = None
        self._parking = None
        self._blackboard = None

    def event(self, event, **values):
        record = {'event': event, 'route': self.current_route, 'unix': time.time(), **values}
        with (self.directory / 'lifecycle_events.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        print('CARLA_LIFECYCLE ' + json.dumps(record), flush=True)

    def load_world(self, evaluator, args, town):
        started = time.monotonic()
        self.event('world_load_started', town=town)
        current = evaluator.client.get_world()
        map_name = current.get_map().name.split('/')[-1]
        moving = [actor.id for actor in current.get_actors()
                  if actor.type_id.startswith(('vehicle.', 'walker.', 'sensor.'))]
        reuse = (self.use_initial_world and self.routes_started == 1
                 and town == self.initial_town == map_name and not moving)
        if reuse:
            world = current
            mode = 'fresh_preloaded_first_episode'
        else:
            world = evaluator.client.load_world(town, reset_settings=False)
            mode = 'full_episode_reload'
        self.event('world_load_finished', town=town, mode=mode,
                   previous_town=map_name, elapsed_seconds=time.monotonic() - started,
                   world_id=world.id, frame=world.get_snapshot().frame)
        return world

    def install(self, namespace):
        import numpy as np
        import torch
        import py_trees
        import leaderboard.utils.parked_vehicles as parked
        from evaluation.carla_static_roads import install_static_road_cache
        self.static_roads = (install_static_road_cache(self.event)
                             if os.environ.get('WMMPC_CACHE_STATIC_ROADS','0')=='1' else None)
        original = namespace['LeaderboardEvaluator']._load_and_run_scenario
        lifecycle = self

        def run_route(evaluator, args, config):
            # Garage/Leaderboard contains process globals that a single-route
            # interpreter would never carry into a later route. In particular,
            # RouteScenario removes parking slots from module-level lists.
            if lifecycle.routes_started == 0:
                lifecycle._python_state = random.getstate()
                lifecycle._numpy_state = np.random.get_state()
                lifecycle._torch_state = torch.random.get_rng_state()
                lifecycle._parking = {name: copy.deepcopy(value) for name, value in vars(parked).items()
                                      if name.startswith('Town') and isinstance(value, list)}
                lifecycle._blackboard = copy.deepcopy(vars(py_trees.blackboard.Blackboard()))
            else:
                random.setstate(lifecycle._python_state)
                np.random.set_state(lifecycle._numpy_state)
                torch.random.set_rng_state(lifecycle._torch_state)
                for name, value in lifecycle._parking.items():
                    setattr(parked, name, copy.deepcopy(value))
                board = vars(py_trees.blackboard.Blackboard())
                board.clear()
                board.update(copy.deepcopy(lifecycle._blackboard))
            lifecycle.routes_started += 1
            lifecycle.current_route = config.name
            lifecycle.event('route_prepare_started', town=config.town,
                            route_sequence=lifecycle.routes_started)
            # Upstream appends +save_name to this value on every route.
            args.agent_config = args.agent_config.split('+')[0]
            started = time.monotonic()
            result = original(evaluator, args, config)
            remaining = []
            try:
                remaining = [{'id': actor.id, 'type': actor.type_id}
                             for actor in evaluator.world.get_actors()
                             if actor.type_id.startswith(('vehicle.', 'walker.', 'sensor.', 'static.prop.', 'static.trigger.'))]
            except RuntimeError as exc:
                lifecycle.event('cleanup_audit_error', error=str(exc))
            lifecycle.event('route_finished', town=config.town, crashed=bool(result),
                            elapsed_seconds=time.monotonic() - started,
                            remaining_route_actors=remaining,
                            next_route_reset='full_episode_reload')
            return result

        namespace['LeaderboardEvaluator']._load_and_run_scenario = run_route
