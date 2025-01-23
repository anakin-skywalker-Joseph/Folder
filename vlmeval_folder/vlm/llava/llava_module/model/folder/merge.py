import math
from typing import Callable, Tuple
import ipdb
import torch


def do_nothing(x, mode=None):
    return x

def merge_wavg_ours(merge: Callable, x: torch.Tensor, size: torch.Tensor = None, metric: torch.Tensor = None, attn_cls: torch.Tensor = None):
    if size is None:
        size = torch.ones_like(x[..., 0, None]) 
    # is_weighted: whether to use weighted merging, is_weighted+is_drop: whether to use direct dropping
    is_weighted = False
    is_drop = False
    x = merge(x * size, mode="sum", is_weighted=is_weighted, token_size=size, is_drop=is_drop) 
    if metric is not None:
        attn_cls = merge(attn_cls.unsqueeze(-1), mode="mean")
        metric = merge(metric * size, mode="sum", is_weighted=is_weighted, token_size=size, is_drop=is_drop)
        if not is_drop:
            size = merge(size, mode="sum")
        else:
            size = merge(size, mode="mean")
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
        # metric = torch.rand((metric.shape)).to(metric.device)
        
        a, b = metric[..., ::2, :], metric[..., 1::2, :]     ## partition
        # a_cls, b_cls = attn_cls[...,::2], attn_cls[...,1::2]   ## (256,98)
        a_cls = attn_cls[...,::2]   ## (256,98)     

        scores_redund = a @ b.transpose(-1, -2)              ## shape=(256,98,98)  
    
        scores = scores_redund - (beta**abs(gamma-num_layer))*alpha*100*a_cls.unsqueeze(-1) #.repeat(1,1,b_cls.shape[-1])
        # mean pooling
        # for i in range(scores.shape[-1]):
        #     scores[:,i,i] = 1000
        if class_token: 
            scores[..., 0, :] = -math.inf        
        node_max, node_idx = scores.max(dim=-1)       ## node_max=(256,98)  node_idx=(256,98)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]       ## edge_idx=(256,98,1)
        unm_idx = edge_idx[..., r:, :] 
        src_idx = edge_idx[..., :r, :]  # r tokens about to be merged
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)      ## dst_idx=(256,r,1)   

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]   
  

    def merge(x: torch.Tensor, mode="mean", is_weighted=False, token_size=None, is_drop=False):
        if not is_weighted and not is_drop:
            src, dst = x[..., ::2, :], x[..., 1::2, :]   ## src.shape=(256,99,768)   dst.shape=(256,98,768)
            n, t1, c = src.shape                         ## 256,99,768
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))    ## unm.shape = (256,98-r,768) 
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))         ## src.shape = (256,r,768)   
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)  ## dst.shape=(256,98,768)   
        else:
            src, dst = x[..., ::2, :], x[..., 1::2, :]   ## src.shape=(256,99,768)   dst.shape=(256,98,768)
            if token_size is None:
                src_attn = src.norm(dim=-1)
                dst_attn = dst.norm(dim=-1)
            else:
                src_attn = (src/token_size[..., ::2, :]).norm(dim=-1)
                dst_attn = (dst/token_size[..., 1::2, :]).norm(dim=-1)   

            n, t1, c = src.shape                         ## 256,99,768
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))    ## unm.shape = (256,98-r,768) 
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))         ## src.shape = (256,r,768)     
            src_attn = src_attn.gather(dim=-1, index=src_idx.squeeze(-1))         ## src.shape = (256,r)   
            dst_attn = dst_attn.gather(dim=-1, index=dst_idx.squeeze(-1))         ## src.shape = (256,r)    
            # max merging
            if is_drop:
                previous_src_attn = src_attn[:]
                src_attn = torch.where(src_attn >= dst_attn, src_attn, torch.zeros_like(src_attn))
                dst_attn = torch.where(dst_attn > previous_src_attn, dst_attn, torch.zeros_like(dst_attn))

            total_attn = src_attn + dst_attn
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), (2*dst_attn/total_attn).unsqueeze(-1).expand(n, r, c), reduce="prod")
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), (2*src_attn/total_attn).unsqueeze(-1)*src, reduce=mode)  ## dst.shape=(256,98,768)
        return torch.cat([unm, dst], dim=1)
    return merge

