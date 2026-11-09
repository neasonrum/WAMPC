"""Build a continuous-observation index and a route-family-isolated dev split."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict, Counter
import gzip
import hashlib
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def family(route):
    # Hold out weather and repeat variants of the same source route together.
    scenario, name = route.split('/')
    m = re.match(r'(Town\d+)_Rep\d+_(.+)_\d+_route(\d+)_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$', name)
    if not m:
        raise ValueError(route)
    return ':'.join((scenario, *m.groups()))


def route_metadata(task):
    rid, name, data_root, output, skip, stride = task
    route = Path(data_root) / name
    frame_ids, poses, speeds, targets = [], [], [], []
    for p in sorted((route / 'measurements').glob('*.json.gz')):
        frame = int(p.name.split('.')[0])
        with gzip.open(p, 'rt') as f:
            m = json.load(f)
        pose = [*m['pos_global'][:2], m['theta']]
        if not np.isfinite(pose).all():
            raise ValueError(f'Nonfinite pose: {p}')
        frame_ids.append(frame)
        poses.append(pose)
        speeds.append(m['speed'])
        targets.append(0.0 if m['brake'] else m['target_speed'])
    frame_ids = np.asarray(frame_ids, dtype=np.int32)
    np.savez(Path(output) / 'metadata' / f'{rid:05}.npz', frames=frame_ids,
             poses=np.asarray(poses, dtype=np.float64), speed=np.asarray(speeds, dtype=np.float32),
             target_speed=np.asarray(targets, dtype=np.float32))
    # Three consecutive past/current observations and >=4 consecutive future observations.
    anchors = [i for i in range(2, len(frame_ids)-4) if frame_ids[i] >= skip
               and frame_ids[i] % stride == 0
               and np.all(np.diff(frame_ids[i-2:i+5]) == 1)]
    return rid, anchors, len(frame_ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=901)
    p.add_argument('--stride', type=int, default=2)
    args = p.parse_args()
    source = ROOT / 'data/full_diffusion/pdm_lite_seed901'
    old = json.loads((source/'manifest.json').read_text())
    routes = json.loads((source/'routes.json').read_text())
    out = args.output.resolve()
    if not out.is_relative_to(ROOT.resolve()) or out.exists():
        raise ValueError('Output must be a NEW directory inside WMMPC')
    out.mkdir(parents=True)
    (out/'metadata').mkdir()
    official = ROOT/'third_party/carla_garage/leaderboard/data/bench2drive220.xml'
    test_triggers = []
    for r in ET.parse(official).getroot().findall('route'):
        for s in r.findall('./scenarios/scenario'):
            t = s.find('trigger_point')
            if t is not None:
                test_triggers.append((r.attrib['town'], s.attrib['type'], float(t.attrib['x']), float(t.attrib['y'])))
    if not test_triggers:
        raise RuntimeError('Official route scenario trigger audit failed')
    template_root = ROOT/'third_party/carla_garage/data'
    template_dirs = {town:list(template_root.glob(f'*_{town}')) for town in ('Town12','Town13')}
    verified, test_families, unknown = set(), set(), []
    for name in routes:
        fam = family(name)
        scenario, town, ident, _ = fam.split(':')
        weather = re.match(r'Town\d+_Rep\d+_.+_(\d+)_route', name.split('/')[1]).group(1)
        candidates = [d/scenario/f'{ident}_{weather}.xml' for d in template_dirs.get(town, [])]
        template = next((x for x in candidates if x.exists()), None)
        if template is None:
            unknown.append(name)
            continue
        rs = ET.parse(template).getroot().find('route')
        verified.add(fam)
        for s in rs.findall('./scenarios/scenario'):
            t = s.find('trigger_point')
            if t is None:
                continue
            x,y = float(t.attrib['x']),float(t.attrib['y'])
            if any(tt==town and st==s.attrib['type'] and (x-tx)**2+(y-ty)**2 < 25
                   for tt,st,tx,ty in test_triggers):
                test_families.add(fam)
    eligible = [r for r in routes if family(r) not in test_families]
    by_scenario = defaultdict(list)
    for r in eligible:
        if family(r) in verified:
            by_scenario[r.split('/')[0]].append(r)
    key = lambda r: hashlib.sha256(f'{args.seed}:{r}'.encode()).hexdigest()
    selected, selected_families = [], set()
    for scenario, names in sorted(by_scenario.items()):
        chosen = min(names, key=key)
        selected.append(chosen)
        selected_families.add(family(chosen))
    remaining = sorted((r for r in eligible if family(r) in verified), key=key)
    for r in remaining:
        if len(selected) >= 44:
            break
        if family(r) not in selected_families:
            selected.append(r)
            selected_families.add(family(r))
    assert len(selected) == 44
    val_names = set(selected)
    train_names = {r for r in eligible if family(r) not in selected_families}
    assert not ({family(r) for r in train_names} & selected_families)
    kept = sorted(train_names | val_names)
    (out/'routes.json').write_text(json.dumps(kept, indent=2)+'\n')
    split = {'seed':args.seed, 'validation_routes':sorted(selected),
             'validation_families':sorted(selected_families),
             'excluded_validation_variants':sorted(set(eligible)-train_names-val_names),
             'excluded_official_test_families':sorted(test_families),
             'official_overlap_rule':'same town/scenario and trigger within 5 m; exclude whole family',
             'unmapped_source_templates':unknown,
             'validation_scenarios':dict(Counter(r.split('/')[0] for r in selected))}
    (out/'split.json').write_text(json.dumps(split, indent=2)+'\n')
    print(json.dumps({'event':'split', 'train_routes':len(train_names), 'val_routes':44,
                     'excluded_test_families':len(test_families),'unmapped_templates':len(unknown)}), flush=True)
    tasks = [(i,r,old['data_root'],str(out),old['skip_first'],args.stride) for i,r in enumerate(kept)]
    train, val, frame_count = [], [], 0
    with ProcessPoolExecutor(args.workers) as pool:
        for count,(rid,anchors,n) in enumerate(pool.map(route_metadata,tasks,chunksize=1),1):
            dest = val if kept[rid] in val_names else train
            dest.extend((rid,i) for i in anchors)
            frame_count += n
            if kept[rid] in val_names and not anchors:
                raise RuntimeError(f'Validation route has no continuous samples: {kept[rid]}')
            if count%100 == 0:
                print(json.dumps({'indexed_routes':count,'total':len(kept),'frames':frame_count}),flush=True)
    np.save(out/'train.npy',np.asarray(train,dtype=np.int32))
    np.save(out/'val.npy',np.asarray(val,dtype=np.int32))
    backbone = ROOT/'ckps/full_diffusion/backbone/all_towns_seed1'
    manifest = {'version':1, 'data_root':old['data_root'], 'routes':len(kept),
                'train_routes':len(train_names), 'validation_routes':44,
                'train_samples':len(train), 'validation_samples':len(val),
                'raw_frames':frame_count, 'sample_stride':args.stride, 'frame_dt_s':0.25,
                'history_frames':3, 'max_future_frames':24, 'query_progress_m':[2.5+i for i in range(10)],
                'path_progress_m':[2.5+i for i in range(10)],
                'teacher_sha256':sha(backbone/'model_0030_1.pth'),
                'teacher_config_sha256':sha(backbone/'config.json'),
                'source_routes_sha256':sha(source/'routes.json'), 'official_xml_sha256':sha(official),
                'split_sha256':sha(out/'split.json'),
                'selection':'route macro average masked multi-step normalized feature MSE',
                'feature_contract':'tfpp-64x64x64-bf16-v1; x forward = width; y right = height; [-32,32] m',
                'temporal_contract':'queries are actual recorded future ego positions/times; no interpolated feature GT'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'event':'complete',**manifest}),flush=True)


if __name__ == '__main__':
    main()
