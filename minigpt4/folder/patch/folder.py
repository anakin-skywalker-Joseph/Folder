from typing import Tuple
import torch
#from timm.models.vision_transformer import Attention, Block, VisionTransformer
import sys
# sys.path.append("../")
# sys.path.append("../../")
# sys.path.append("../../models")
from minigpt4.models.eva_vit import Attention, Block, VisionTransformer
from minigpt4.folder.merge import bipartite_unimodal_matching, merge_wavg_ours
from minigpt4.folder.utils import parse_r
import ipdb
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from minigpt4.models.eva_vit import Mlp
import torch.nn.functional as F

class FolderAttention(Attention):
    def forward(self, x: torch.Tensor, size: torch.Tensor = None, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn_cls = torch.mean(attn[...,0,:],dim=1)
        # attn_cls = torch.mean(attn,dim=[1,2])
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)                           

        return x, k.mean(1), attn_cls

class FolderBlock(Block):                                          ## 整个模块中插入 Folder Inference 模块
    def forward(self, x: torch.Tensor, rel_pos_bias=None):   ## x.shape=[256,197,768]
        attn_size = self._folder_info["size"] if self._folder_info["prop_attn"] else None

        if self.gamma_1 is None:
            x_attn, metric, attn_cls = self.attn(self.norm1(x), attn_size, rel_pos_bias=rel_pos_bias)      ## metric.shape=[256, 197, 64]     x_attn.shape=[256,197,768]       
            x = x + self.drop_path(x_attn)
            r = self._folder_info["r"].pop(0)
            if r > (x.shape[-2]-1)//2: #fold
                r_remove = min(x.shape[-2]//2-1,r)
                r = r-r_remove
                while r_remove > 0:
                    merge = bipartite_unimodal_matching(metric, 
                                                        attn_cls,
                                                        r_remove,
                                                        self._folder_info["class_token"], 
                                                        self._folder_info["alpha"],
                                                        self._folder_info["num_layer"],
                                                        self._folder_info["beta"],
                                                        self._folder_info["gamma"],
                                                        self._folder_info["r_threshold"])
                    x, self._folder_info["size"], metric, attn_cls = merge_wavg_ours(merge, x, self._folder_info["size"], metric, attn_cls)
                    r_remove = min((x.shape[-2]-1)//2,r)
                    r = r-r_remove
            elif r > 0:
                merge = bipartite_unimodal_matching(metric, 
                                                    attn_cls,
                                                    r,
                                                    self._folder_info["class_token"], 
                                                    self._folder_info["alpha"],
                                                    self._folder_info["num_layer"],
                                                    self._folder_info["beta"],
                                                    self._folder_info["gamma"],
                                                    self._folder_info["r_threshold"])
                x, self._folder_info["size"] = merge_wavg_ours(merge, x, self._folder_info["size"])
            self._folder_info["num_layer"]+=1     
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x_attn, metric, attn_cls = self.attn(self.norm1(x), attn_size, rel_pos_bias=rel_pos_bias)      ## metric.shape=[256, 197, 64]     x_attn.shape=[256,197,768]       
            x = x + self.drop_path(self.gamma_1 * x_attn)
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
                                                        self._folder_info["beta"],
                                                        self._folder_info["gamma"],
                                                        self._folder_info["r_threshold"])
                    x, self._folder_info["size"], metric, attn_cls = merge_wavg_ours(merge, x, self._folder_info["size"], metric, attn_cls)
                    r_remove = min((x.shape[-2]-1)//2,r)
                    r = r-r_remove
            elif r > 0:
                merge = bipartite_unimodal_matching(metric, 
                                                    attn_cls,
                                                    r,
                                                    self._folder_info["class_token"], 
                                                    self._folder_info["alpha"],
                                                    self._folder_info["num_layer"],
                                                    self._folder_info["beta"],
                                                    self._folder_info["gamma"],
                                                    self._folder_info["r_threshold"])
                x, self._folder_info["size"] = merge_wavg_ours(merge, x, self._folder_info["size"])
            self._folder_info["num_layer"]+=1
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


def make_folder_class(transformer_class):
    class FolderVisionTransformer(transformer_class):    
        def forward(self, *args, **kwdargs) -> torch.Tensor:
            if self.is_turbo:
                self._folder_info["r"] = parse_r(len(self.blocks), self.r)
                if (len(self.blocks)*self.r)%4!=0:
                    self._folder_info["r"][-1] += (4-(len(self.blocks)*self.r)%4)
            else:
                removed_token = self.r*len(self.blocks)
                if removed_token%4!=0:
                    removed_token += (4-removed_token%4)
                self._folder_info["r"] = [0 for _ in range(len(self.blocks)-3)]+[0, 0, removed_token]
            self._folder_info["size"] = None
            self._folder_info["source"] = None
            self._folder_info["alpha"] = self.alpha
            self._folder_info["beta"] = self.beta
            self._folder_info["gamma"] =self.gamma
            self._folder_info["num_layer"]=self.num_layer
            self._folder_info["r_threshold"] =self.r_threshold
            return super().forward(*args, **kwdargs)

        def generate(self, samples,*args, **kwdargs):
            if self.is_turbo:
                self._folder_info["r"] = parse_r(len(self.blocks), self.r)
                if (len(self.blocks)*self.r)%4!=0:
                    self._folder_info["r"][-1] += (4-(len(self.blocks)*self.r)%4)
            else:
                removed_token = self.r*len(self.blocks)
                if removed_token%4!=0:
                    removed_token += (4-removed_token%4)
                self._folder_info["r"] = [0 for _ in range(len(self.blocks)-3)]+[0, 0, removed_token]
            self._folder_info_visual["size"] = None
            self._folder_info_visual["source"] = None
            self._folder_info["alpha"] = self.alpha
            self._folder_info["beta"] = self.beta
            self._folder_info["gamma"] =self.gamma
            self._folder_info["num_layer"]=self.num_layer
            self._folder_info["r_threshold"] =self.r_threshold
            return super().generate(samples,*args, **kwdargs)

        def forward_image(self, *args, **kwdargs) -> torch.Tensor:
            if self.is_turbo:
                self._folder_info["r"] = parse_r(len(self.blocks), self.r)
                if (len(self.blocks)*self.r)%4!=0:
                    self._folder_info["r"][-1] += (4-(len(self.blocks)*self.r)%4)
            else:
                removed_token = self.r*len(self.blocks)
                if removed_token%4!=0:
                    removed_token += (4-removed_token%4)
                self._folder_info["r"] = [0 for _ in range(len(self.blocks)-3)]+[0, 0, removed_token]
            self._folder_info_visual["size"] = None
            self._folder_info_visual["source"] = None
            self._folder_info["alpha"] = self.alpha
            self._folder_info["beta"] = self.beta
            self._folder_info["gamma"] =self.gamma
            self._folder_info["num_layer"]=self.num_layer
            self._folder_info["r_threshold"] =self.r_threshold
            return super().forward_image(*args, **kwdargs)

        def forward_text(self, *args, **kwdargs) -> torch.Tensor:
            if self.is_turbo:
                self._folder_info["r"] = parse_r(len(self.blocks), self.r)
                if (len(self.blocks)*self.r)%4!=0:
                    self._folder_info["r"][-1] += (4-(len(self.blocks)*self.r)%4)
            else:
                removed_token = self.r*len(self.blocks)
                if removed_token%4!=0:
                    removed_token += (4-removed_token%4)
                self._folder_info["r"] = [0 for _ in range(len(self.blocks)-3)]+[0, 0, removed_token]
            self._folder_info_visual["size"] = None
            self._folder_info_visual["source"] = None
            self._folder_info["alpha"] = self.alpha
            self._folder_info["beta"] = self.beta
            self._folder_info["gamma"] =self.gamma
            self._folder_info["num_layer"]=self.num_layer
            self._folder_info["r_threshold"] =self.r_threshold
            return super().forward_text(*args, **kwdargs)
    return FolderVisionTransformer

def apply_patch(model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True):
    FolderVisionTransformer = make_folder_class(model.__class__)    
    model.__class__ = FolderVisionTransformer
    model.r = 0 
    model.alpha=1
    model.beta=1   #控制衰减底数
    model.gamma=0  #控制衰减指数
    model.num_layer=0
    model.r_threshold = 40
    model.is_turbo = False
    model._folder_info = {"r": model.r,"size": None,"source": None,"trace_source": trace_source,"prop_attn": prop_attn,
        "class_token": model.cls_token is not None, "alpha":model.alpha,
        "beta":model.beta,"gamma":model.gamma,"num_layer":0,"r_threshold" :model.r_threshold}

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = FolderBlock
            module._folder_info = model._folder_info
        elif isinstance(module, Attention):
            module.__class__ = FolderAttention




