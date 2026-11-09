import torch
import torch.nn.functional as F


def error_metrics(prediction,target,spatial_valid,query_valid,corridor):
    prediction,target=prediction.float(),target.float()
    mask=spatial_valid.float()*query_valid[:,:,None,None,None]
    channels=target.shape[2]
    error=(prediction-target).square().mean(2,keepdim=True)
    energy=target.square().mean(2,keepdim=True)
    count=mask.sum((-1,-2,-3)).clamp_min(1.)
    mse=(error*mask).sum((-1,-2,-3))/count
    power=(energy*mask).sum((-1,-2,-3))/count
    nmse=mse/power.clamp_min(1e-4)
    corridor_mask=mask*corridor[:,None]
    corridor_count=corridor_mask.sum((-1,-2,-3)).clamp_min(1.)
    corridor_error=(error*corridor_mask).sum((-1,-2,-3))/corridor_count
    corridor_power=(energy*corridor_mask).sum((-1,-2,-3))/corridor_count
    cosine=F.cosine_similarity(prediction,target,dim=2,eps=1e-6)[:,:,None]
    nonzero=mask*(energy>1e-8)
    cosine=(cosine*nonzero).sum((-1,-2,-3))/nonzero.sum((-1,-2,-3)).clamp_min(1.)
    return {'nmse':nmse,'mse':mse,'corridor_nmse':corridor_error/corridor_power.clamp_min(1e-4),
            'cosine':cosine}


def training_loss(metrics,valid):
    loss=metrics['nmse']+.5*metrics['corridor_nmse']+.05*(1-metrics['cosine'])
    # Equal sequence weighting; unavailable geometric-progress targets are excluded.
    return ((loss*valid).sum(1)/valid.sum(1).clamp_min(1)).mean()
