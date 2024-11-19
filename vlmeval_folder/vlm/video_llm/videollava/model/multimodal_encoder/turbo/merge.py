import math
from typing import Callable, Tuple
import ipdb
import torch


def do_nothing(x, mode=None):
    return x


def merge_wavg(merge: Callable, x: torch.Tensor, size: torch.Tensor = None):
    ## 所有token的特征 x.shape=(256,197,768)
    if size is None:
        size = torch.ones_like(x[..., 0, None])  
    x = merge(x * size, mode="sum")  ## x.shape=(256,197-r,768)
    size = merge(size, mode="sum")   ## size.shape=(256,197-r,1)
    x = x / size                     
    return x, size


def bipartite_soft_matching(metric: torch.Tensor, r: int,class_token: bool = False,r_threshold=0):
    protected = 0
    if class_token:
        protected += 1

    t = metric.shape[1]                ## metric.shape=(256,197,64) 
    r = min(r, (t - protected) // 2)   ## 最多token只裁剪到 196//2

    if r <= 0 or t<r_threshold:                         
        return do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)  
        a, b = metric[..., ::2, :], metric[..., 1::2, :]     ## token分两份
        
        scores = a @ b.transpose(-1, -2)                     ## shape=(256,99,98)  两份算出相似度矩阵

        if class_token:
            scores[..., 0, :] = -math.inf

        ## 对于第0列，找其中每个元素在第1列中最相似的 token
        node_max, node_idx = scores.max(dim=-1)                          ## node_max=(256,99)  node_idx=(256,99)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]  ## edge_idx=(256,99,1)  

        unm_idx = edge_idx[..., r:, :]  # 除了相似度最高的r个
        src_idx = edge_idx[..., :r, :]  # 要被合并的r个tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)      ## dst_idx=(256,r,1)   
        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]   

    def merge(x: torch.Tensor, mode="mean"):
        src, dst = x[..., ::2, :], x[..., 1::2, :]   ## src.shape=(256,99,768)   dst.shape=(256,98,768)
        n, t1, c = src.shape                         ## 256,99,768
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))    ## unm.shape = (256,99-r,768)  第0列要保存的token编码
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))         ## src.shape = (256,r,768)     第0列要删除的token编码
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)  ## dst.shape=(256,98,768)   第1列的token编码

        return torch.cat([unm, dst], dim=1)    

    return merge


def merge_wavg_ours(merge: Callable, x: torch.Tensor, size: torch.Tensor = None, metric: torch.Tensor = None, attn_cls: torch.Tensor = None):
    # if size is None:
    #     size = torch.ones_like(x[..., 0, None])     
    # x_all = merge(x[:,1:,:]*size[:,1:,:], mode="sum")  ## x.shape=(256,197-r,768)
    # size_all = merge(size[:,1:,:], mode="sum")   ## size.shape=(256,197-r,1)
    # x_all = x_all / size_all
    # x = torch.cat([x[:,0:1,:], x_all], dim=1)
    # size = torch.cat([size[:,0:1,:], size_all], dim=1)
    # return x, size
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

    t = metric.shape[1]                ## metric.shape=(256,197,64) 
    r = min(r, (t - protected) // 2)   ## 最多token只裁剪到 196//2

    if r <= 0 or t<r_threshold:                         
        return do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)  

        # metric = metric[:,1:,:]      ## (256,196,64)  
        
        a, b = metric[..., ::2, :], metric[..., 1::2, :]     ## token分两份，每一份都是 (256,98,64)
        # a_cls, b_cls = attn_cls[...,::2], attn_cls[...,1::2]   ## (256,98)
        a_cls = attn_cls[...,::2]   ## (256,98)     
        # scores=torch.rand((a.shape[-2],b.shape[-1])).to(a.device)
        scores_redund = a @ b.transpose(-1, -2)              ## shape=(256,98,98)  token冗余度
     
        # if gamma==0 and beta==0:
        #     # scores = torch.rand(scores_redund.shape).to(scores_redund.device)
        #     scores=scores_redund/(a_cls.unsqueeze(-1).repeat(1,1,b_cls.shape[-1])+0.001)
        # elif gamma==0 and beta==2:
        #     scores=scores_redund*(1-alpha*30*a_cls.unsqueeze(-1).repeat(1,1,b_cls.shape[-1]))
        # else:
        #     scores = scores_redund - (beta**abs(gamma-num_layer))*alpha*30*a_cls.unsqueeze(-1).repeat(1,1,b_cls.shape[-1])
        scores = scores_redund - (beta**abs(gamma-num_layer))*alpha*100*a_cls.unsqueeze(-1) #.repeat(1,1,b_cls.shape[-1])
        if class_token: 
            scores[..., 0, :] = -math.inf        
        node_max, node_idx = scores.max(dim=-1)       ## node_max=(256,98)  node_idx=(256,98)
        
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]       ## edge_idx=(256,98,1)  降序排列
        unm_idx = edge_idx[..., r:, :]  # 除了相似度最高的r个
        src_idx = edge_idx[..., :r, :]  # 要被合并的r个tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)      ## dst_idx=(256,r,1)   

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]   

    def merge(x: torch.Tensor, mode="mean"):
        src, dst = x[..., ::2, :], x[..., 1::2, :]   ## src.shape=(256,99,768)   dst.shape=(256,98,768)
        n, t1, c = src.shape                         ## 256,99,768
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))    ## unm.shape = (256,98-r,768)  第0列要保存的token编码
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))         ## src.shape = (256,r,768)     第0列要删除的token编码
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)  ## dst.shape=(256,98,768)   第1列的token编码
        return torch.cat([unm, dst], dim=1)    

    return merge

