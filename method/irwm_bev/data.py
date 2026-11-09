"""Continuous raw observations; geometric actions and exact observed feature targets."""
from functools import lru_cache
import json
import os
from pathlib import Path
import cv2
import laspy
import numpy as np
import torch
from torch.utils.data import Dataset, default_collate
from method.full_diffusion.data import read_measurement
from method.full_diffusion.garage import load_config, make_preprocessor, BACKBONE


class SequenceDataset(Dataset):
    def __init__(self,index,split,cache_root):
        self.index=Path(index)
        self.manifest=json.loads((self.index/'manifest.json').read_text())
        self.routes=json.loads((self.index/'routes.json').read_text())
        self.samples=np.load(self.index/f'{split}.npy',mmap_mode='r')
        self.data_root=Path(self.manifest['data_root'])
        identity=self.manifest['teacher_sha256'][:16]+'_'+self.manifest['teacher_config_sha256'][:12]+'_bf16_v1'
        self.cache_root=Path(cache_root)/identity
        self.processor=None

    def __len__(self):
        return len(self.samples)

    @lru_cache(maxsize=16)
    def metadata(self,rid):
        with np.load(self.index/'metadata'/f'{rid:05}.npz') as f:
            return {k:f[k].copy() for k in f.files}

    def sensor(self,rid,frame):
        # Route name, not split-dependent integer ID, identifies cached content.
        name=self.routes[rid]
        cache=self.cache_root/name/f'{frame:04}.npy'
        if cache.exists():
            bits=np.load(cache,allow_pickle=False)
            if bits.shape!=(64,64,64) or bits.dtype!=np.uint16:
                raise ValueError(f'Invalid cache {cache}')
            return {'key':str(cache),'bev':torch.from_numpy(bits).view(torch.bfloat16)}
        if (self.cache_root/'complete.json').exists():
            raise FileNotFoundError(f'Completed teacher cache is missing {cache}; refusing raw dataset fallback')
        if self.processor is None:
            cv2.setNumThreads(0)
            self.processor=make_preprocessor(load_config(BACKBONE),training=False)
        import transfuser_utils
        route=self.data_root/name
        m=read_measurement(route/'measurements'/f'{frame:04}.json.gz')
        image=cv2.imread(str(route/'rgb'/f'{frame:04}.jpg'),cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f'Cannot decode RGB {route}/{frame}')
        image=transfuser_utils.crop_array(self.processor.config,cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        cloud=laspy.read(route/'lidar'/f'{frame:04}.laz').xyz
        cloud=self.processor.align(cloud,m,m)
        lidar=self.processor.lidar_to_histogram_features(cloud,self.processor.config.use_ground_plane)
        return {'key':str(cache),'rgb':torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))),
                'lidar':torch.from_numpy(np.ascontiguousarray(lidar))}

    def __getitem__(self,index):
        rid,anchor=map(int,self.samples[index])
        m=self.metadata(rid)
        frames=m['frames']
        last=min(anchor+24,len(frames)-1)
        for i in range(anchor+1,last+1):
            if frames[i]!=frames[i-1]+1:
                last=i-1
                break
        pose=m['poses'][anchor]
        c,s=np.cos(pose[2]),np.sin(pose[2])
        rotation=np.array([[c,-s],[s,c]])
        xy=(m['poses'][anchor:last+1,:2]-pose[:2])@rotation
        progress=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
        times=np.arange(len(xy),dtype=np.float32)*.25
        query_indices=[]
        qvalid=[]
        if progress[-1]<2.5:
            # Dwell/slow examples still contain evolving actors. Actual observed frames only.
            query_indices=np.linspace(1,len(xy)-1,10).round().astype(int).tolist()
            qvalid=np.r_[True,np.diff(query_indices)>0].tolist()
        else:
            previous=0
            for distance in self.manifest['query_progress_m']:
                nearest=int(np.argmin(abs(progress[1:]-distance)))+1
                valid=(nearest>previous and progress[-1]>=distance-.5)
                if valid:
                    previous=nearest
                query_indices.append(previous if previous else 1)
                qvalid.append(valid)
        q=np.asarray(query_indices,dtype=np.int64)
        qmask=np.asarray(qvalid,dtype=bool)
        assert qmask.any()
        # Interpolate ONLY action coordinates/times along the observed polyline, never features.
        unique=np.r_[True,np.diff(progress)>1e-5]
        path_progress=np.arange(10,dtype=np.float32)+2.5
        path=np.stack([np.interp(path_progress,progress[unique],xy[unique,j]) for j in range(2)],-1)
        path_times=np.interp(path_progress,progress[unique],times[unique])
        path_valid=path_progress<=progress[-1]
        indices=np.r_[np.arange(anchor-2,anchor+1),anchor+q]
        source_poses=m['poses'][indices]
        translations=(source_poses[:,:2]-pose[:2])@rotation
        yaw=source_poses[:,2]-pose[2]
        relative=np.c_[translations,np.arctan2(np.sin(yaw),np.cos(yaw))].astype(np.float32)
        return {'sensors':[self.sensor(rid,int(frames[i])) for i in indices],
                'source_in_anchor':torch.from_numpy(relative),
                'path':torch.tensor(path,dtype=torch.float32),'path_times':torch.tensor(path_times,dtype=torch.float32),
                'path_valid':torch.from_numpy(path_valid), 'query_xy':torch.tensor(xy[q],dtype=torch.float32),
                'query_times':torch.from_numpy(times[q]),'query_valid':torch.from_numpy(qmask),
                'speed':torch.tensor(m['speed'][anchor]),'target_speed':torch.tensor(m['target_speed'][anchor]),
                'route_id':rid,'anchor_frame':int(frames[anchor])}


def collate(samples):
    sensors=[s['sensors'] for s in samples]
    return {**default_collate([{k:v for k,v in s.items() if k!='sensors'} for s in samples]),'sensors':sensors}


class TeacherCache:
    def __init__(self,perception,device,chunk=4):
        self.perception=perception
        self.device=device
        self.chunk=chunk
        self.hits=0
        self.misses=0

    @torch.no_grad()
    def __call__(self,sensors):
        flat=[f for row in sensors for f in row]
        resolved={f['key']:f['bev'] for f in flat if 'bev' in f}
        missing={f['key']:f for f in flat if f['key'] not in resolved}
        values=list(missing.values())
        self.hits+=len(flat)-len(values)
        self.misses+=len(values)
        for start in range(0,len(values),self.chunk):
            group=values[start:start+self.chunk]
            padded=group+[group[-1]]*(self.chunk-len(group))
            rgb=torch.stack([f['rgb'] for f in padded]).to(self.device,non_blocking=True)
            lidar=torch.stack([f['lidar'] for f in padded]).to(self.device,non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                bev,_,_=self.perception.net.backbone(rgb.float(),lidar.float())
            if bev.shape[1:]!=(64,64,64) or not torch.isfinite(bev).all():
                raise RuntimeError(f'Invalid teacher features {bev.shape}')
            bev=bev[:len(group)].to(torch.bfloat16).cpu()
            for f,value in zip(group,bev):
                value=value.contiguous()
                resolved[f['key']]=value
                path=Path(f['key'])
                path.parent.mkdir(parents=True,exist_ok=True)
                temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
                with temporary.open('wb') as handle:
                    np.save(handle,value.view(torch.uint16).numpy(),allow_pickle=False)
                os.replace(temporary,path)
        # Fresh and cached values both pass through the same bf16 representation.
        return torch.stack([resolved[f['key']] for f in flat]).to(self.device,non_blocking=True).reshape(len(sensors),len(sensors[0]),64,64,64)
