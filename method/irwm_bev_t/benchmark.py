"""Bounded two-GPU throughput comparison at the SAME optimizer batch and weights."""
import argparse
from datetime import timedelta, datetime, timezone
import gc
import json
import os
from pathlib import Path
import random
import statistics
import time
import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, BatchSampler
from .data import SequenceDataset, TeacherCache
from .model import ActionBEVWorldModel
from .train import loader_for, RemainingBatches, update, prepare
from .launch import ROOT, guard_context, memory_snapshot, write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=80)
    p.add_argument('--warmup',type=int,default=15)
    args=p.parse_args()
    guard_context('13053006')
    rank,local,world=(int(os.environ[k]) for k in ('RANK','LOCAL_RANK','WORLD_SIZE'))
    assert world==2
    torch.set_num_threads(1)
    torch.cuda.set_device(local)
    device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(minutes=5))
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    base=checkpoint['config']
    train=SequenceDataset(ROOT/base['index'],'train',ROOT/base['cache'])
    teacher=TeacherCache(device)
    epoch=checkpoint['progress']['epoch']
    samples_consumed=checkpoint['progress']['next_batch']*base['batch_per_gpu']
    choices=[(4,4,2,2),(8,2,2,2),(16,1,2,2),(16,1,3,2),(16,1,3,4),(4,4,2,2)]
    reports=[]
    args.output.mkdir(parents=True,exist_ok=True)
    for run,(batch,accumulation,workers,prefetch) in enumerate(choices):
        assert world*batch*accumulation==32 and world*(workers+1)<=len(os.sched_getaffinity(0))
        torch.manual_seed(base['seed']+rank)
        np.random.seed(base['seed']+rank)
        random.seed(base['seed']+rank)
        config={**base,'batch_per_gpu':batch,'accumulation_steps':accumulation,
                'workers_per_rank':workers,'prefetch_factor':prefetch}
        sampler=DistributedSampler(train,num_replicas=world,rank=rank,seed=base['seed'],drop_last=True)
        sampler.set_epoch(epoch)
        batches=BatchSampler(sampler,batch,drop_last=True)
        offset=samples_consumed//batch
        required=(args.steps+args.warmup)*accumulation
        if offset+required>len(batches):
            offset=0
        loader=loader_for(train,config,batch_sampler=RemainingBatches(batches,offset,offset+required),
                          generator=torch.Generator().manual_seed(base['seed']+epoch*world+rank))
        model=ActionBEVWorldModel(**config['model']).to(device)
        model.load_state_dict(checkpoint['model'],strict=True)
        policy=DDP(model,device_ids=[local],broadcast_buffers=False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=base['learning_rate'],weight_decay=base['weight_decay'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Benchmark-only scheduler does not touch the saved training schedule.
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        iterator=iter(loader)
        torch.cuda.reset_peak_memory_stats()
        timings=[]
        loads=[]
        last_loss=None
        dist.barrier()
        start=time.perf_counter()
        for step in range(args.steps+args.warmup):
            torch.cuda.synchronize()
            t=time.perf_counter()
            microbatches=[next(iterator) for _ in range(accumulation)]
            load_elapsed=time.perf_counter()-t
            loss,norm=update(microbatches,teacher,device,policy,optimizer,scheduler)
            del microbatches
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-t
            if step==0:
                first_loss=float(loss)
                first_weights=torch.cat([v.detach().flatten() for v in model.parameters()]).cpu()
                if run==0:
                    reference_first=first_weights
                    first_error=0.
                else:
                    first_error=float((first_weights-reference_first).abs().max())
                del first_weights
            if step>=args.warmup:
                timings.append(elapsed)
                loads.append(load_elapsed)
            last_loss=float(loss)
            if step%10==0 and memory_snapshot().get('unreclaimable_fraction',1)>.5:
                raise RuntimeError('Benchmark RAM guard: refusing further tuning')
        del iterator,loader
        local_report={'rank':rank,'seconds':time.perf_counter()-start,'mean_update_seconds':statistics.mean(timings),
                      'median_update_seconds':statistics.median(timings),'p95_update_seconds':float(np.quantile(timings,.95)),
                      'mean_loader_wait_seconds':statistics.mean(loads),'first_loss':first_loss,'last_loss':last_loss,
                      'first_update_max_parameter_difference':first_error,
                      'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                      'peak_gpu_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'memory':memory_snapshot()}
        gathered=[None]*world
        dist.all_gather_object(gathered,local_report)
        report={'run':run,'batch_per_gpu':batch,'accumulation_steps':accumulation,'workers_per_rank':workers,
                'prefetch_factor':prefetch,'effective_batch':32,'warmup_updates':args.warmup,'measured_updates':args.steps,
                'max_rank_mean_seconds':max(r['mean_update_seconds'] for r in gathered),'ranks':gathered,
                'checkpoint_updates':checkpoint['progress']['updates'],'epoch':epoch+1,'rank_sample_offset':offset*batch,
                'utc':datetime.now(timezone.utc).isoformat()}
        reports.append(report)
        if rank==0:
            write_json(args.output/f'case_{run:02}.json',report)
            print(json.dumps(report),flush=True)
        del policy,optimizer,scheduler,model
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
    if rank==0:
        # Require a meaningful improvement against BOTH baseline measurements.
        baseline=min(reports[0]['max_rank_mean_seconds'],reports[-1]['max_rank_mean_seconds'])
        best=min(reports[1:-1],key=lambda x:x['max_rank_mean_seconds'])
        result={'baseline_seconds':baseline,'best':best,'speedup':baseline/best['max_rank_mean_seconds'],
                'adopt':best['max_rank_mean_seconds']<baseline*.95,'all_cases':reports}
        write_json(args.output/'result.json',result)
        print(json.dumps({'event':'benchmark_complete','speedup':result['speedup'],'adopt':result['adopt'],
                          'batch':best['batch_per_gpu'],'workers':best['workers_per_rank'],'prefetch':best['prefetch_factor']}),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':
    main()
