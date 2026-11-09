"""Geometry, valid-mask and bf16-cache contract tests used before GPU acceptance."""
import torch
from .geometry import metric_grid,align_to_anchor
from .loss import error_metrics


def geometry_tests(device='cpu'):
    grid=metric_grid(device)
    source=grid.permute(2,0,1)[None]
    aligned,valid=align_to_anchor(source,torch.zeros(1,3,device=device))
    torch.testing.assert_close(aligned,source,rtol=0,atol=1e-5)
    assert valid.all()
    translated,mask=align_to_anchor(source,torch.tensor([[3.,0.,0.]],device=device))
    translated_valid=mask.clone()
    assert (~translated_valid).any()
    expected=source-torch.tensor([3.,0.],device=device)[None,:,None,None]
    torch.testing.assert_close(translated[mask.expand_as(translated)],expected[mask.expand_as(expected)],rtol=0,atol=1e-5)
    rotated,mask=align_to_anchor(source,torch.tensor([[0.,0.,torch.pi/2]],device=device))
    expected=torch.stack((source[:,1],-source[:,0]),1)
    torch.testing.assert_close(rotated[mask.expand_as(rotated)],expected[mask.expand_as(expected)],rtol=0,atol=1e-4)
    value=torch.randn(64,64,64).bfloat16()
    restored=torch.from_numpy(value.view(torch.uint16).numpy().copy()).view(torch.bfloat16)
    assert torch.equal(value,restored)
    target=torch.ones(1,1,2,64,64,device=device)
    mask=translated_valid[:,None]
    pred=target.clone()
    pred[~mask.expand_as(pred)]=1e4
    valid=torch.ones(1,1,dtype=torch.bool,device=device)
    corridor=torch.ones(1,1,64,64,device=device)
    assert error_metrics(pred,target,mask,valid,corridor)['nmse'].item()==0
    print('Geometry and feature-cache contract tests passed',flush=True)



def interface_tests(dataset,device='cpu'):
    import json
    import numpy as np
    from types import SimpleNamespace
    from .launch import ROOT
    from .data import motion_targets
    from .interface import positions_from_deltas,deltas_from_positions,from_v3_normalized,from_v3_policy,rebase_positions
    from method.full_diffusionv3.diffusion import SpatialTemporalCodec
    from method.full_diffusionv3.data import future_motion_targets
    settings=json.loads((ROOT/'method/full_diffusionv3/config.json').read_text())['labels']
    max_difference=0.
    for i in (0,13,131,min(1009,len(dataset)-1)):
        rid,anchor=map(int,dataset.samples[i])
        m=dataset.metadata(rid)
        measurement=lambda j: {'pos_global':m['poses'][j,:2].tolist(),'theta':float(m['poses'][j,2]),'speed':float(m['speed'][j])}
        expected=future_motion_targets(measurement(anchor),{k:measurement(anchor+k) for k in range(2,13,2)},settings,False)
        delta,positions,_,_=motion_targets(m['poses'],anchor)
        np.testing.assert_allclose(delta,expected['temporal_delta'],atol=2e-5,rtol=1e-5)
        np.testing.assert_allclose(positions,expected['future_positions'],atol=2e-5,rtol=1e-5)
        max_difference=max(max_difference,float(np.abs(delta-expected['temporal_delta']).max()))
    mean=np.linspace(-1,1,32).reshape(16,2).tolist()
    std=np.linspace(1,5,32).reshape(16,2).tolist()
    codec=SpatialTemporalCodec(mean=mean,std=std).to(device)
    delta=torch.tensor([[[1.,.1],[2.,.3],[3.,-.1],[2.,.4],[1.,.3],[.5,-.2]]],device=device)
    spatial=torch.full((1,10,2),999.,device=device)
    decoded=from_v3_normalized(codec.encode(spatial,delta),codec)
    torch.testing.assert_close(decoded,delta,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(from_v3_policy(SimpleNamespace(last_temporal_delta=delta)),delta)
    positions=positions_from_deltas(delta)
    torch.testing.assert_close(deltas_from_positions(positions),delta,rtol=1e-5,atol=1e-6)
    identity=torch.eye(3,device=device)[None]
    current=torch.tensor([[[0.,-1.,2.],[1.,0.,3.],[0.,0.,1.]]],device=device)
    rebased=rebase_positions(positions,identity,current)
    restored=rebase_positions(rebased,current,identity)
    torch.testing.assert_close(restored,positions,rtol=1e-5,atol=1e-6)
    reconstructed=positions_from_deltas(deltas_from_positions(rebased))
    torch.testing.assert_close(reconstructed,rebased,rtol=1e-5,atol=1e-6)
    # Numerically verify mean-loss accumulation against a single effective batch.
    a=torch.nn.Linear(2,1,bias=False,dtype=torch.float64)
    b=torch.nn.Linear(2,1,bias=False,dtype=torch.float64)
    b.load_state_dict(a.state_dict())
    x=torch.arange(64,dtype=torch.float64).reshape(32,2)/64
    y=x[:,0:1]*2-x[:,1:2]
    ((a(x)-y)**2).mean().backward()
    for xx,yy in zip(x.chunk(4),y.chunk(4)):
        (((b(xx)-yy)**2).mean()/4).backward()
    torch.testing.assert_close(a.weight.grad,b.weight.grad,rtol=1e-12,atol=1e-12)
    return {'v3_label_max_error_metres':max_difference,'metric_decode':True,
            'pose_round_trip':True,'increment_round_trip':True,'mean_gradient_accumulation':True}
