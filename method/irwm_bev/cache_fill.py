"""Complete the immutable frozen-teacher feature cache on healthy allocated nodes."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from method.full_diffusion.garage import FrozenGaragePerception
from .data import SequenceDataset,TeacherCache
from .launch import ROOT,guard_context,write_json


class MissingFrames(Dataset):
    def __init__(self,tasks,shard,shards):
        self.source=SequenceDataset(ROOT/'data/irwm_bev/pdm_seed901_dev44_v1','train',ROOT/'data/irwm_bev/teacher_cache')
        self.tasks=np.load(tasks,mmap_mode='r')[shard::shards]
    def __len__(self):
        return len(self.tasks)
    def __getitem__(self,i):
        route,frame=map(int,self.tasks[i])
        return self.source.sensor(route,frame)


def collate_frames(values):
    return [[v] for v in values]


def scan(output,final=False):
    ds=SequenceDataset(ROOT/'data/irwm_bev/pdm_seed901_dev44_v1','train',ROOT/'data/irwm_bev/teacher_cache')
    missing=[]
    total=0
    for rid,route in enumerate(ds.routes):
        frames=ds.metadata(rid)['frames']
        # Every selected anchor is >=10 and has two contiguous history frames.
        frames=frames[frames>=8]
        total+=len(frames)
        directory=ds.cache_root/route
        existing={int(p.name[:-4]) for p in os.scandir(directory) if p.name.endswith('.npy')} if directory.exists() else set()
        missing.extend((rid,int(frame)) for frame in frames if int(frame) not in existing)
        if (rid+1)%1000==0:
            print(json.dumps({'scanned_routes':rid+1,'missing':len(missing),'required':total}),flush=True)
    report={'required_frames':total,'missing_frames':len(missing),'routes':len(ds.routes),
            'teacher_sha256':ds.manifest['teacher_sha256'],'cache_root':str(ds.cache_root),
            'completed_unix':time.time()}
    if final:
        if missing:
            raise RuntimeError(f'Cache still misses {len(missing)} frames')
        write_json(ds.cache_root/'complete.json',report)
        write_json(output/'cache_complete.json',report)
    else:
        np.save(output/'missing.npy',np.asarray(missing,dtype=np.int32).reshape(-1,2))
        write_json(output/'scan.json',report)
    print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--job',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scan',action='store_true')
    p.add_argument('--verify',action='store_true')
    p.add_argument('--node-index',type=int)
    args=p.parse_args()
    guard_context(args.job)
    args.output=args.output.resolve()
    if not args.output.is_relative_to(ROOT):
        raise ValueError('Output must remain inside WMMPC')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.scan or args.verify:
        scan(args.output,final=args.verify)
        return
    local=int(os.environ['LOCAL_RANK'])
    shard=args.node_index*2+local
    torch.set_num_threads(1)
    torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    device=torch.device('cuda',local)
    ds=MissingFrames(args.output/'missing.npy',shard,6)
    loader=DataLoader(ds,batch_size=16,num_workers=2,collate_fn=collate_frames,
                      multiprocessing_context='spawn',prefetch_factor=1,pin_memory=True,timeout=120)
    teacher=TeacherCache(FrozenGaragePerception().to(device),device,chunk=4)
    start=time.monotonic()
    done=0
    for i,sensors in enumerate(loader):
        teacher(sensors)
        done+=len(sensors)
        if i%20==0:
            info={'shard':shard,'done':done,'total':len(ds),'seconds':time.monotonic()-start}
            write_json(args.output/f'shard{shard}.progress.json',info)
            print(json.dumps(info),flush=True)
    write_json(args.output/f'shard{shard}.complete.json',{'shard':shard,'frames':done,'seconds':time.monotonic()-start})


if __name__=='__main__':
    main()
