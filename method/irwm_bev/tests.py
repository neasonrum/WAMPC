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


if __name__=='__main__':
    geometry_tests()
