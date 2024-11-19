from typing import Tuple
import torch
import sys
sys.path.append("../../")
sys.path.append("../../models")
from models.vit import Attention, Block, VisionTransformer
from folder.merge import bipartite_unimodal_matching, merge_folder
from folder.utils import parse_r
import torch.nn as nn
from timm.models.layers import DropPath
from models.vit import Mlp

class folderAttention(Attention):
    def forward(self, x: torch.Tensor, size: torch.Tensor = None):
        B, N, C = x.shape  
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4))
        q, k, v = (qkv[0], qkv[1], qkv[2])  
        attn = (q @ k.transpose(-2, -1)) * self.scale 
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)  
        attn_cls=torch.mean(attn[...,0,:],dim=1)  
        attn = self.attn_drop(attn)  
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  
        x = self.proj(x)                                
        x = self.proj_drop(x)                            
        return x, k.mean(1),attn_cls


class folderBlock(Block):                                          
    def _drop_path1(self, x):   
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)
    def forward(self, x: torch.Tensor,*args): 
        attn_size = self._folder_info["size"] if self._folder_info["prop_attn"] else None
        x_attn, metric, attn_cls = self.attn(self.norm1(x), attn_size)                    
        x = x + x_attn                                            
        # metric = x[:]
        r = self._folder_info["r"].pop(0)
        if r > (x.shape[-2]-1)//2: #fold
            r_remove = min((x.shape[-2]-1)//2,r)
            r = r-r_remove
            while r_remove > 0:
                merge = bipartite_unimodal_matching(metric, 
                                                    attn_cls,
                                                    r_remove,
                                                    self._folder_info["class_token"], 
                                                    self._folder_info["alpha"],
                                                    self._folder_info["num_layer"],
                                                    self._folder_info["r_threshold"])
                x, self._folder_info["size"], metric, attn_cls = merge_folder(merge, x, self._folder_info["size"], metric, attn_cls)
                r_remove = min((x.shape[-2]-1)//2,r)
                r = r-r_remove

        elif r > 0:
            merge = bipartite_unimodal_matching(metric, 
                                                attn_cls,
                                                r,
                                                self._folder_info["class_token"], 
                                                self._folder_info["alpha"],
                                                self._folder_info["num_layer"],
                                                self._folder_info["r_threshold"])
            x, self._folder_info["size"] = merge_folder(merge, x, self._folder_info["size"])
        self._folder_info["num_layer"]+=1         
        x = x + self.mlp(self.norm2(x))                                    
        return x


def make_folder_class(transformer_class):
    class folderVisionTransformer(transformer_class):    
        def forward(self, *args, **kwdargs) -> torch.Tensor:
            if self.is_turbo:
                self._folder_info["r"] = parse_r(len(self.blocks), self.r)
            else:
                removed_token = self.r*len(self.blocks)
                remove_proportion = [self.beta,self.gamma,10]
                print("remove proportion: ",remove_proportion)
                print("total removed token: ",removed_token)
                last_remove = int(removed_token*(remove_proportion[2]/sum(remove_proportion)))
                second_remove = int(removed_token*(remove_proportion[1]/sum(remove_proportion)))
                first_remove = removed_token-last_remove-second_remove 
                self._folder_info["r"] = [0 for _ in range(len(self.blocks)-3)]+[first_remove, second_remove, last_remove]
            self._folder_info["size"] = None
            self._folder_info["source"] = None
            self._folder_info["alpha"] = self.alpha
            self._folder_info["beta"] = self.beta
            self._folder_info["gamma"] =self.gamma
            self._folder_info["num_layer"]=self.num_layer
            self._folder_info["r_threshold"] =self.r_threshold
            return super().forward(*args, **kwdargs)

    return folderVisionTransformer

def apply_patch(model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True):
    folderVisionTransformer = make_folder_class(model.__class__)    
    model.__class__ = folderVisionTransformer
    model.r = 0 
    model.alpha=1
    model.beta=1   
    model.gamma=0  
    model.num_layer=0
    model.r_threshold = 40
    model.is_turbo = False
    model._folder_info = {"r": model.r,"size": None,"source": None,"trace_source": trace_source,"prop_attn": prop_attn,
        "class_token": model.cls_token is not None, "alpha":model.alpha,
        "beta":model.beta,"gamma":model.gamma,"num_layer":0,"r_threshold" :model.r_threshold}

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = folderBlock
            module._folder_info = model._folder_info
        elif isinstance(module, Attention):
            module.__class__ = folderAttention




