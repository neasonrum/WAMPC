"""Reproduce the epoch-boundary data-worker import path without a simulator mount."""
import json
from pathlib import Path
import sys
from torch.utils.data import DataLoader,Subset
from method.full_diffusion.garage import load_config,make_preprocessor
from .data import SequenceDataset,collate


def main():
    root=Path(__file__).resolve().parents[2]
    make_preprocessor(load_config(),training=False)
    from agents.navigation import global_route_planner
    assert str(root/'third_party/carla_navigation_0_9_15') in global_route_planner.__file__
    assert not any('/datasets/public/carla/0.9.15/PythonAPI/carla' in p for p in sys.path)
    dataset=SequenceDataset(root/'data/irwm_bev/pdm_seed901_dev44_v1','val',root/'data/irwm_bev/teacher_cache')
    sizes=[]
    for _ in range(2):
        loader=DataLoader(Subset(dataset,range(4)),batch_size=4,num_workers=2,
                          multiprocessing_context='spawn',timeout=60,collate_fn=collate)
        for batch in loader:
            sizes.append(list(batch['path'].shape))
    print(json.dumps({'passed':True,'spawned_loader_passes':len(sizes),'path_shapes':sizes,
                      'navigation':global_route_planner.__file__}),flush=True)


if __name__=='__main__':
    main()
