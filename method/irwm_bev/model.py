"""Native PyTorch adaptation of IR-WM's action-conditioned residual BEV evolution.

No OCC labels/head, future camera inputs, or future pose rotations enter this model.
The 10-point path and queried executed/candidate XY+time are explicit actions.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from .geometry import metric_grid


class Residual(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.net=nn.Sequential(nn.GroupNorm(8,channels),nn.SiLU(),nn.Conv2d(channels,channels,3,padding=1),
                               nn.GroupNorm(8,channels),nn.SiLU(),nn.Conv2d(channels,channels,3,padding=1))
    def forward(self,x):
        return x+self.net(x)


class ActionBEVWorldModel(nn.Module):
    def __init__(self,hidden=96,history=3,path_nodes=10):
        super().__init__()
        self.history=history
        self.path_nodes=path_nodes
        self.hidden=hidden
        self.history_encoder=nn.Sequential(nn.Conv2d(history*65+2,hidden,5,stride=2,padding=2),
                                           Residual(hidden),Residual(hidden))
        self.path_input=nn.Linear(6,hidden)
        self.path_position=nn.Parameter(torch.randn(1,path_nodes+1,hidden)*.01)
        layer=nn.TransformerEncoderLayer(hidden,4,hidden*4,dropout=0.,batch_first=True,norm_first=True)
        self.path_encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.cross=nn.MultiheadAttention(hidden,4,dropout=0.,batch_first=True)
        self.cross_norm=nn.LayerNorm(hidden)
        self.action=nn.Sequential(nn.Linear(8,hidden*2),nn.SiLU(),nn.Linear(hidden*2,hidden*2))
        self.state_input=nn.Conv2d(64,hidden,3,stride=2,padding=1)
        self.gates=nn.Conv2d(hidden*2,hidden*2,3,padding=1)
        self.candidate=nn.Sequential(nn.Conv2d(hidden*2,hidden,3,padding=1),nn.GroupNorm(8,hidden),nn.Tanh())
        self.transition=Residual(hidden)
        self.decode=nn.Sequential(nn.Conv2d(hidden,hidden,3,padding=1),nn.SiLU(),nn.Conv2d(hidden,64,1))
        nn.init.zeros_(self.decode[-1].weight)
        nn.init.zeros_(self.decode[-1].bias)

    def forward(self,history,history_valid,path,path_times,path_valid,query_xy,query_times,query_valid,speed,target_speed):
        b,nh,c,h,w=history.shape
        if (nh,c,h,w)!=(self.history,64,64,64) or path.shape[1:]!=(self.path_nodes,2):
            raise ValueError('Wrong TF++ feature or path shape')
        spatial=metric_grid(history.device).permute(2,0,1)[None].expand(b,-1,-1,-1)/32.
        visible=torch.cat((history*history_valid,history_valid.float()),2).flatten(1,2)
        state=self.history_encoder(torch.cat((visible,spatial),1))
        origin=path.new_zeros(b,1,2)
        nodes=torch.cat((origin,path),1)
        deltas=torch.cat((origin,nodes[:,1:]-nodes[:,:-1]),1)
        times=torch.cat((path_times.new_zeros(b,1),path_times),1)
        valid=torch.cat((torch.ones(b,1,dtype=torch.bool,device=path.device),path_valid),1)
        inputs=torch.cat((nodes/32.,deltas/5.,times[...,None]/6.,valid[...,None].float()),-1)
        # PyTorch 2.5's cuDNN SDPA cannot plan the short masked path-token shape on A100.
        # Scope the backend selection to this world model, including inference use.
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.MATH]):
            tokens=self.path_encoder(self.path_input(inputs)+self.path_position,src_key_padding_mask=~valid)
            flat=state.flatten(2).transpose(1,2)
            context,_=self.cross(self.cross_norm(flat),tokens,tokens,key_padding_mask=~valid,need_weights=False)
        state=state+context.transpose(1,2).reshape_as(state)
        predicted=history[:,-1]
        outputs=[]
        previous_xy=query_xy.new_zeros(b,2)
        previous_time=query_times.new_zeros(b)
        for k in range(query_xy.shape[1]):
            dt=(query_times[:,k]-previous_time).clamp_min(0.)
            action=torch.cat((query_xy[:,k]/32.,(query_xy[:,k]-previous_xy)/5.,
                              query_times[:,k,None]/6.,dt[:,None]/2.,speed[:,None]/15.,target_speed[:,None]/15.),1)
            scale,shift=self.action(action).chunk(2,1)
            encoded=self.state_input(predicted)
            encoded=encoded*(1+.1*scale[:,:,None,None].tanh())+shift[:,:,None,None]
            reset,update=self.gates(torch.cat((state,encoded),1)).sigmoid().chunk(2,1)
            proposal=self.candidate(torch.cat((reset*state,encoded),1))
            next_state=self.transition((1-update)*state+update*proposal)
            delta=self.decode(F.interpolate(next_state,size=(64,64),mode='bilinear',align_corners=False))
            next_prediction=predicted+delta*dt.clamp(max=4.)[:,None,None,None]
            keep=query_valid[:,k,None,None,None]
            state=torch.where(keep,next_state,state)
            predicted=torch.where(keep,next_prediction,predicted)
            outputs.append(predicted)
            previous_xy=torch.where(query_valid[:,k,None],query_xy[:,k],previous_xy)
            previous_time=torch.where(query_valid[:,k],query_times[:,k],previous_time)
        return torch.stack(outputs,1)

    @torch.no_grad()
    def rollout(self,history,history_valid,path,speed,target_speed,arrival_times=None):
        """Roll out all 10 planner nodes. Supply elapsed arrival times when available.

        Default timing is only a constant-average-speed estimate, not a controller model.
        Repeated XY with increasing arrival_times represents dwell at that location.
        """
        if arrival_times is None:
            velocity=.5*(speed+target_speed)
            if (velocity<.5).any():
                raise ValueError('For stopped/slow trajectories supply explicit arrival_times')
            distances=torch.cat((path[:,:1].norm(dim=-1),(path[:,1:]-path[:,:-1]).norm(dim=-1)),1)
            arrival_times=distances.cumsum(1)/velocity[:,None]
        if (arrival_times<0).any() or (arrival_times[:,1:]<arrival_times[:,:-1]).any():
            raise ValueError('Arrival times must be nonnegative and monotonic')
        if (arrival_times>6.).any():
            raise ValueError('Requested arrival exceeds the 6 s training observation window')
        valid=torch.ones(path.shape[:2],device=path.device,dtype=torch.bool)
        return self(history,history_valid,path,arrival_times,valid,path,arrival_times,valid,speed,target_speed)
