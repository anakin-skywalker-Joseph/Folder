import math
from typing import Callable, Tuple
import torch


def do_nothing(x, mode=None):
    return x

def merge_folder(merge: Callable, x: torch.Tensor, size: torch.Tensor = None, metric: torch.Tensor = None, attn_cls: torch.Tensor = None):
    if size is None:
        size = torch.ones_like(x[..., 0, None])  
    x = merge(x * size, mode="sum",is_weighted=False, token_size=None)
    # x = merge(x, mode="sum")
    if metric is not None:
        attn_cls = merge(attn_cls.unsqueeze(-1), mode="sum")
        metric = merge(metric * size, mode="sum",is_weighted=False, token_size=None)
        size = merge(size, mode="sum")
        x = x / size                  
        metric = metric / size   
        return x, size, metric, attn_cls.squeeze(-1)
    else:
        size = merge(size, mode="sum")
        x = x / size
        return x, size


def bipartite_unimodal_matching(metric:torch.Tensor, attn_cls:torch.Tensor, r:int, class_token:bool=False,alpha=1,num_layer=0,r_threshold=0):
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
        a_cls = attn_cls[...,::2]       
        scores_redund = a @ b.transpose(-1, -2)              
        scores = scores_redund - alpha*30*a_cls.unsqueeze(-1)
        if class_token: 
            scores[..., 0, :] = -math.inf        
        node_max, node_idx = scores.max(dim=-1)     
        
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]  

        unm_idx = edge_idx[..., r:, :]  
        src_idx = edge_idx[..., :r, :]  
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)     
        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]   

    def merge(x: torch.Tensor, mode="mean", is_weighted=False, token_size=None, is_drop=False):
        if not is_weighted:
            src, dst = x[..., ::2, :], x[..., 1::2, :]   ## src.shape=(256,99,768)   dst.shape=(256,98,768)
            n, t1, c = src.shape                         ## 256,99,768
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))    
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))         
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)  
        else:
            src, dst = x[..., ::2, :], x[..., 1::2, :]   
            # src_attn, dst_attn = attn_cls[..., ::2], attn_cls[..., 1::2]
            if token_size is None:
                src_attn = src.norm(dim=-1)
                dst_attn = dst.norm(dim=-1)
            else:
                src_attn = (src / token_size[..., ::2, :]).norm(dim=-1)
                dst_attn = (dst / token_size[..., 1::2, :]).norm(dim=-1)
            n, t1, c = src.shape                        
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))    
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))         
            src_attn = src_attn.gather(dim=-1, index=src_idx.squeeze(-1))         
            dst_attn = dst_attn.gather(dim=-1, index=dst_idx.squeeze(-1))         
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
