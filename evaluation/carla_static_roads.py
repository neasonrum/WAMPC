"""Reuse immutable OpenDRIVE graphs; keep a fresh route-decision state each time."""
import copy
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import time

PLANNER_SHA256='9b164a2de723d91d55cd1203543747580c1c6c509ed0294a6de17b001f7db8d0'
ROOT=Path(__file__).resolve().parents[1]


class RoadPickler(pickle.Pickler):
    def persistent_id(self,obj):
        import carla
        if isinstance(obj,carla.Map):return ('map',obj.name)
        if isinstance(obj,carla.Waypoint):
            transform=obj.transform;loc=transform.location;rot=transform.rotation
            return ('waypoint',obj.road_id,obj.section_id,obj.lane_id,obj.s,
                    (loc.x,loc.y,loc.z,rot.pitch,rot.yaw,rot.roll))
        return None


class RoadUnpickler(pickle.Unpickler):
    def __init__(self,handle,world_map):
        super().__init__(handle);self.world_map=world_map;self.waypoints={};self.topology=None

    def persistent_load(self,key):
        if key[0]=='map':
            if key[1]!=self.world_map.name:raise ValueError('Cached map name differs')
            return self.world_map
        if key[0]!='waypoint':raise ValueError('Unknown road cache reference')
        _,road,section,lane,distance,expected=key
        identity=(road,section,lane,distance)
        if identity not in self.waypoints:
            def matches(waypoint):
                if waypoint is None or (waypoint.road_id,waypoint.section_id,waypoint.lane_id,waypoint.s)!=identity:
                    return False
                transform=waypoint.transform;loc=transform.location;rot=transform.rotation
                return (loc.x,loc.y,loc.z,rot.pitch,rot.yaw,rot.roll)==expected
            waypoint=self.world_map.get_waypoint_xodr(road,lane,distance)
            # get_waypoint_xodr narrows s to float32 in this API. Restore the
            # original double using native next/previous, then require exact
            # equality of the complete waypoint identity and transform.
            for _ in range(3):
                if matches(waypoint) or waypoint is None:break
                delta=distance-waypoint.s
                if not delta:break
                options=waypoint.next(delta) if delta>0 else waypoint.previous(-delta)
                options=[w for w in options if w.road_id==road and w.lane_id==lane]
                if not options:break
                waypoint=min(options,key=lambda w:abs(w.s-distance))
            if not matches(waypoint):
                # Topological endpoints can be on a section boundary where the
                # xodr query selects another lane section. Obtain their original
                # native waypoint directly; no coordinate or s approximation.
                if self.topology is None:
                    self.topology={}
                    for pair in self.world_map.get_topology():
                        for w in pair:self.topology[(w.road_id,w.section_id,w.lane_id,w.s)]=w
                waypoint=self.topology.get(identity)
            if not matches(waypoint):raise ValueError(f'Cannot exactly restore waypoint {identity}')
            self.waypoints[identity]=waypoint
        return self.waypoints[identity]


class StaticRoadGraphCache:
    def __init__(self, constructor, event, cache_dir=None):
        self.constructor=constructor
        self.event=event
        self.templates={}
        self.map_identity=None
        self.cache_dir=Path(cache_dir).resolve() if cache_dir is not None else None
        if self.cache_dir:
            self.cache_dir.relative_to(ROOT.resolve())
            self.cache_dir.mkdir(parents=True,exist_ok=True)

    def disk_template(self,world_map,resolution,identity):
        import networkx,numpy
        import carla
        version={'format':1,'map':identity,'resolution':resolution,'source':PLANNER_SHA256,
                 'networkx':networkx.__version__,'numpy':numpy.__version__,
                 'python':list(__import__('sys').version_info[:2])}
        key=hashlib.sha256(json.dumps(version,sort_keys=True).encode()).hexdigest()
        path=self.cache_dir/(key+'.pickle');metadata=self.cache_dir/(key+'.json')
        with (self.cache_dir/(key+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if path.exists() and metadata.exists():
                try:
                    info=json.loads(metadata.read_text())
                    if hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha256']:
                        raise ValueError('Road cache checksum differs')
                    with path.open('rb') as handle:
                        planner=RoadUnpickler(handle,world_map).load()
                    self.event('static_road_disk_cache',cache_hit=True,path=str(path),bytes=path.stat().st_size)
                    return planner
                except (OSError,ValueError,pickle.UnpicklingError,EOFError,RuntimeError) as exc:
                    self.event('static_road_disk_cache_rejected',path=str(path),error=repr(exc))
            planner=self.constructor(world_map,resolution)
            temporary=path.with_suffix(f'.{os.getpid()}.pickle.tmp')
            with temporary.open('wb') as handle:RoadPickler(handle,protocol=5).dump(planner)
            # Reject a cache that cannot reproduce every waypoint's transform
            # exactly; use the ordinary freshly constructed planner regardless.
            try:
                with temporary.open('rb') as handle:RoadUnpickler(handle,world_map).load()
            except (ValueError,RuntimeError,pickle.UnpicklingError) as exc:
                self.event('static_road_disk_cache_unusable',error=repr(exc),path=str(temporary))
                return planner
            temporary.replace(path)
            info={'identity':version,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'bytes':path.stat().st_size,'unix':time.time()}
            temporary_metadata=metadata.with_suffix(f'.{os.getpid()}.json.tmp')
            temporary_metadata.write_text(json.dumps(info,indent=2)+'\n');temporary_metadata.replace(metadata)
            self.event('static_road_disk_cache',cache_hit=False,path=str(path),bytes=path.stat().st_size)
            return planner

    def __call__(self, world_map, sampling_resolution):
        started=time.monotonic()
        identity=(world_map.name,hashlib.sha256(world_map.to_opendrive().encode()).hexdigest())
        # Retain one town, with its separate 1m/2m graphs. This bounds memory
        # instead of accumulating twelve large-town maps in every worker.
        if identity!=self.map_identity:
            self.templates.clear()
            self.map_identity=identity
        resolution=float(sampling_resolution)
        hit=resolution in self.templates
        if not hit:
            self.templates[resolution]=(self.disk_template(world_map,resolution,identity) if self.cache_dir
                                        else self.constructor(world_map,sampling_resolution))
        # The pinned planner only changes these per-instance scalar decision
        # fields while tracing. Its topology, graph, waypoints, id maps and
        # OpenDRIVE map are read-only after construction. The pristine template
        # is never returned or used to trace a route.
        planner=copy.copy(self.templates[resolution])
        self.event('static_road_graph',map_name=world_map.name,map_sha256=identity[1],
                   sampling_resolution=resolution,cache_hit=hit,
                   elapsed_seconds=time.monotonic()-started)
        return planner


def install_static_road_cache(event):
    import srunner.scenariomanager.carla_data_provider as provider
    import leaderboard.utils.route_manipulation as manipulation
    original=provider.GlobalRoutePlanner
    if original is not manipulation.GlobalRoutePlanner:
        raise RuntimeError('Expected a shared original GlobalRoutePlanner class')
    source=Path(inspect.getfile(original))
    if hashlib.sha256(source.read_bytes()).hexdigest()!=PLANNER_SHA256:
        raise RuntimeError('CARLA road planner changed; review its mutable state before caching')
    cached=StaticRoadGraphCache(original,event,ROOT/'.runtime/static_road_graphs_v1')
    provider.GlobalRoutePlanner=cached
    manipulation.GlobalRoutePlanner=cached
    return cached
