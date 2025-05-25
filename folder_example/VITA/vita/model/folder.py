import math
from typing import Callable, Tuple
import torch


def transform_index(a, b):
    merged = torch.cat((a,b), dim=1)
    sorted_value, sorted_indices = torch.sort(merged, dim=1)
    rank_indices = torch.argsort(sorted_indices, dim=1)
    a_ranked = rank_indices[:, :a.shape[1]]
    b_ranked = rank_indices[:, a.shape[1]:]
    return a_ranked, b_ranked

def do_nothing(x, mode=None):
    return x

def merge_folder(merge: Callable, x: torch.Tensor, size: torch.Tensor = None, metric: torch.Tensor = None, attn_cls: torch.Tensor = None):
    if size is None:
        size = torch.ones_like(x[..., 0, None])  
    x = merge(x * size, mode="sum",is_weighted=False, token_size=None)
    # x = merge(x, mode="sum")
    if metric is not None:
        attn_cls = merge(attn_cls.unsqueeze(-1), mode="sum").squeeze(-1) if attn_cls is not None else None
        metric = merge(metric * size, mode="sum",is_weighted=False, token_size=None)
        size = merge(size, mode="sum")
        x = x / size                  
        metric = metric / size   
        return x, size, metric, attn_cls
    else:
        size = merge(size, mode="sum")
        x = x / size
        return x, size


def bipartite_unimodal_matching(metric:torch.Tensor, r:int, attn_cls:torch.Tensor=None, class_token:bool=False, alpha=1, r_threshold=0):
    protected = 0
    if class_token:
        protected += 1

    t = metric.shape[-2]               
    r = min(r, (t - protected) // 2)  

    if r <= 0 or t<r_threshold:                         
        return do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)  
        a, b = metric[..., ::2, :], metric[..., 1::2, :]           
        scores = a @ b.transpose(-1, -2)  
        if attn_cls is not None: 
            a_cls = attn_cls[...,::2]            
            scores = scores - alpha*30*a_cls.unsqueeze(-1)
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
        if not is_weighted: # average merging
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
        
        if unm_idx.shape[-2] == 0:
            return torch.cat([unm, dst], dim=1)
        t2 = dst.shape[1]
        unm_ranked, dst_ranked = transform_index(2*unm_idx.squeeze(2), torch.arange(1, 2*t2+1, 2, device=x.device).view(1, t2).expand(n, t2))

        final_tensor = torch.zeros((n, unm_idx.shape[-2]+t2, c), device=x.device).to(x.dtype)
        final_tensor.scatter_(1,unm_ranked.unsqueeze(-1).repeat(1,1,c),unm)
        final_tensor.scatter_(1,dst_ranked.unsqueeze(-1).repeat(1,1,c),dst)
        return final_tensor

    return merge

def merge_features(image_features, metric=None, size=None, r=1, class_token=True):
    is_batch = True
    if len(image_features.shape) == 2:
        is_batch = False
        image_features = image_features.unsqueeze(0)
    if metric is None:
        metric = image_features[:]
    if r > (image_features.shape[-2]-1)//2: #fold
        r_remove = min((image_features.shape[-2]-1)//2,r)
        r = r-r_remove
        while r_remove > 0:
            merge = bipartite_unimodal_matching(metric=metric, 
                                                r=r_remove,
                                                attn_cls=None,
                                                class_token=class_token,
                                                )
            image_features, size, metric, _ = merge_folder(merge, image_features, size, metric)
            r_remove = min((image_features.shape[-2]-1)//2,r)
            r = r-r_remove
    elif r > 0:
        merge = bipartite_unimodal_matching(metric=metric, 
                                            r=r,
                                            attn_cls=None,
                                            class_token=class_token,
                                            )
        image_features, size, metric, _ = merge_folder(merge, image_features, size, metric)
    if r < image_features.shape[-2] // 2:
        log_size = 1 + size.log()
        image_features = image_features * log_size
    if is_batch == False:
        image_features = image_features.squeeze(0)
    return image_features, size, metric
