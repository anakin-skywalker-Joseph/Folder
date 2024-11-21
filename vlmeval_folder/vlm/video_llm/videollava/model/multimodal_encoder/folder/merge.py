import math
from typing import Callable, Tuple
import ipdb
import torch


def do_nothing(x, mode=None):
    return x

def merge_wavg_ours(merge: Callable, x: torch.Tensor, size: torch.Tensor = None, metric: torch.Tensor = None, attn_cls: torch.Tensor = None):
    if size is None:
        size = torch.ones_like(x[..., 0, None]) 
    x = merge(x * size, mode="sum") 
    if metric is not None:
        attn_cls = merge(attn_cls.unsqueeze(-1), mode="mean")
        metric = merge(metric * size, mode="sum")
        size = merge(size, mode="sum")
        x = x / size                  
        metric = metric / size   
        return x, size, metric, attn_cls.squeeze(-1)
    else:
        size = merge(size, mode="sum")
        x = x / size
        return x, size


def bipartite_unimodal_matching(metric:torch.Tensor, attn_cls:torch.Tensor, r:int, class_token:bool=False,alpha=1,num_layer=0,beta=1,gamma=0,r_threshold=0):
    protected = 0
    if class_token:
        protected += 1

    t = metric.shape[1]                
    r = min(r, (t - protected) // 2)   

    if r <= 0 or t<r_threshold:                         
        return do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)  
        
        a, b = metric[..., ::2, :], metric[..., 1::2, :]     
        
        a_cls = attn_cls[...,::2]   ## (256,98)     
        scores_redund = a @ b.transpose(-1, -2)             
        scores = scores_redund - (beta**abs(gamma-num_layer))*alpha*100*a_cls.unsqueeze(-1)
        if class_token: 
            scores[..., 0, :] = -math.inf        
        node_max, node_idx = scores.max(dim=-1)       
        
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]       
        unm_idx = edge_idx[..., r:, :]  
        src_idx = edge_idx[..., :r, :] 
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)      ## dst_idx=(256,r,1)   

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]   

    def merge(x: torch.Tensor, mode="mean"):
        src, dst = x[..., ::2, :], x[..., 1::2, :]   ## src.shape=(256,99,768)   dst.shape=(256,98,768)
        n, t1, c = src.shape                         ## 256,99,768
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))    ## unm.shape = (256,98-r,768)  
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))         ## src.shape = (256,r,768)     
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)  ## dst.shape=(256,98,768)
        return torch.cat([unm, dst], dim=1)    

    return merge

