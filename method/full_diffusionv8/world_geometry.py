import torch
import torch.nn.functional as F


def metric_grid(device, dtype=torch.float32, size=64, extent=32.):
    centers = (torch.arange(size, device=device, dtype=dtype)+.5)*(2*extent/size)-extent
    y,x = torch.meshgrid(centers,centers,indexing='ij')
    return torch.stack((x,y),-1)


def align_to_anchor(features, source_in_anchor, extent=32.):
    """Pull source-frame BEV into the current ego frame. Poses are target construction only.

    features: [N,C,H,W]; source_in_anchor: [N,3] (x,y,yaw).
    TF++ histogram layout: width=x forward, height=y right.
    """
    n,c,h,w = features.shape
    if h != w:
        raise ValueError('Only square metric BEV is supported')
    grid = metric_grid(features.device, size=h, extent=extent)[None]
    q = grid - source_in_anchor[:,None,None,:2].float()
    cosine = source_in_anchor[:,2].float().cos()[:,None,None]
    sine = source_in_anchor[:,2].float().sin()[:,None,None]
    source = torch.stack((cosine*q[...,0]+sine*q[...,1],
                          -sine*q[...,0]+cosine*q[...,1]),-1)
    # Full interpolation footprint must be inside the source feature field.
    limit = extent-extent/h
    valid = (source.abs() <= limit+1e-5).all(-1)[:,None]
    aligned = F.grid_sample(features.float(),source/extent,mode='bilinear',
                            padding_mode='zeros',align_corners=False)
    return aligned,valid


def path_corridor(path, path_valid, queries, query_valid, radius=3.):
    """Distance to action polyline, in the current ego frame."""
    b = len(path)
    grid = metric_grid(path.device)[None,None]
    origin = path.new_zeros(b,1,2)
    points = torch.cat((origin,path),1)
    valid = torch.cat((torch.ones(b,1,device=path.device,dtype=torch.bool),path_valid),1)
    start,end = points[:,:-1],points[:,1:]
    segment = (end-start)[:,:,None,None]
    offset = grid-start[:,:,None,None]
    ratio = (offset*segment).sum(-1)/segment.square().sum(-1).clamp_min(1e-8)
    distance = (offset-ratio.clamp(0,1)[...,None]*segment).square().sum(-1)
    distance = distance.masked_fill(~(valid[:,:-1]&valid[:,1:])[:,:,None,None],float('inf'))
    node_dist = (grid-queries[:,:,None,None]).square().sum(-1)
    node_dist = node_dist.masked_fill(~query_valid[:,:,None,None],float('inf'))
    ego_dist = grid[:,0].square().sum(-1).expand(b,-1,-1)
    closest = torch.minimum(torch.minimum(distance.min(1).values,node_dist.min(1).values),ego_dist)
    return (closest <= radius**2)[:,None]

