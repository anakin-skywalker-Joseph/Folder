from typing import Tuple
import torch
import sys
sys.path.append("../../")
sys.path.append("../../models")
from models.vit_retrieval import Attention, Block, VisionTransformer
from turbo.merge import bipartite_soft_matching, merge_wavg, bipartite_unimodal_matching,merge_wavg_ours
from turbo.utils import parse_r
import torch.nn as nn
from timm.models.layers import DropPath
from models.vit import Mlp
class TurboBlock_raw(Block):                                          
    def _drop_path1(self, x):   
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor,*args):   ## x.shape=[256,197,768]
        attn_size = self._turbo_info["size"] if self._turbo_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size, is_tome=True)      ## metric.shape=[256, 197, 64]     x_attn.shape=[256,197,768]
        x = x + self._drop_path1(x_attn)                          ## x.shape = [256, 197, 768]
        r = self._turbo_info["r"].pop(0)

        if r > 0:
            merge = bipartite_soft_matching(metric, r, self._turbo_info["class_token"],self._turbo_info["r_threshold"])
            x, self._turbo_info["size"] = merge_wavg(merge, x, self._turbo_info["size"])  ## token合并
            
        x = x + self._drop_path2(self.mlp(self.norm2(x)))         ## x.shape = [256,197-r=133,768]
        return x


def make_turbo_class_raw(transformer_class):
    class TurboVisionTransformer_raw(transformer_class):    
        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._turbo_info["r"] = parse_r(len(self.blocks), self.r)
            self._turbo_info["size"] = None
            self._turbo_info["source"] = None
            self._turbo_info["r_threshold"] =self.r_threshold
            return super().forward(*args, **kwdargs)

    return TurboVisionTransformer_raw

class TurboBlock(Block):                                         
    def _drop_path1(self, x):   
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)
    def forward(self, x: torch.Tensor,*args):   ## x.shape=[256,197,768]
        attn_size = self._turbo_info["size"] if self._turbo_info["prop_attn"] else None
        x_attn, metric, attn_cls = self.attn(self.norm1(x), attn_size,is_turbo=True)      ## metric.shape=[256, 197, 64]     x_attn.shape=[256,197,768]                     
        x = x + x_attn                                            ## x.shape = [256, 197, 768]
        r = self._turbo_info["r"].pop(0)
        if r > 0:  
            merge = bipartite_unimodal_matching(metric, 
                                                attn_cls,
                                                r,
                                                self._turbo_info["class_token"], 
                                                self._turbo_info["alpha"],
                                                self._turbo_info["num_layer"],
                                                self._turbo_info["beta"],
                                                self._turbo_info["gamma"],
                                                self._turbo_info["r_threshold"])
            x, self._turbo_info["size"] = merge_wavg_ours(merge, x, self._turbo_info["size"])            
            self._turbo_info["num_layer"]+=1           
        x = x + self.mlp(self.norm2(x))                                    ## x.shape = [256,197-r=181,768]
        return x


def make_turbo_class(transformer_class):
    class TurboVisionTransformer(transformer_class):    
        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._turbo_info["r"] = parse_r(len(self.blocks), self.r)
            self._turbo_info["size"] = None
            self._turbo_info["source"] = None
            self._turbo_info["alpha"] = self.alpha
            self._turbo_info["beta"] = self.beta
            self._turbo_info["gamma"] =self.gamma
            self._turbo_info["num_layer"]=self.num_layer
            self._turbo_info["r_threshold"] =self.r_threshold
            return super().forward(*args, **kwdargs)

    return TurboVisionTransformer

def apply_patch(model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True):
    TurboVisionTransformer = make_turbo_class(model.__class__)    
    model.__class__ = TurboVisionTransformer
    model.r = 0 
    model.alpha=1
    model.beta=1   
    model.gamma=0 
    model.num_layer=0
    model.r_threshold = 0
    model._turbo_info = {"r": model.r,"size": None,"source": None,"trace_source": trace_source,"prop_attn": prop_attn,
        "class_token": model.cls_token is not None, "alpha":model.alpha,
        "beta":model.beta,"gamma":model.gamma,"num_layer":0,"r_threshold" :model.r_threshold}

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = TurboBlock
            module._turbo_info = model._turbo_info

def apply_patch_tome(model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True):
    TurboVisionTransformer_raw = make_turbo_class_raw(model.__class__)    
    model.__class__ = TurboVisionTransformer_raw
    model.r = 0 
    model.alpha=1
    model.r_threshold = 0
    model._turbo_info = {"r": model.r,"size": None,"source": None,"trace_source": trace_source,"prop_attn": prop_attn,
        "class_token": model.cls_token is not None,"alpha":model.alpha,"r_threshold":model.r_threshold}

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = TurboBlock_raw
            module._turbo_info = model._turbo_info




