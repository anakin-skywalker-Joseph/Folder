from typing import Tuple, Optional
import torch
import sys
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPAttention, CLIPEncoderLayer, CLIPVisionTransformer
from ..merge import bipartite_unimodal_matching, merge_wavg_ours
from ..utils import parse_r
import ipdb
import torch.nn as nn
import torch.nn.functional as F

class FolderAttention(CLIPAttention):
    def forward(self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        size: Optional[torch.Tensor] = None):
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if size is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights + size.log()[:, None, None, :, 0]
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_cls = torch.mean(attn_weights.reshape(bsz, self.num_heads, tgt_len, src_len)[...,0,:],dim=1)
        
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, key_states.reshape(bsz, self.num_heads, -1, self.head_dim).mean(1), attn_cls

class FolderEncoderLayer(CLIPEncoderLayer):                                         
    def forward(self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
        ):   ## x.shape=[256,197,768]
        attn_size = self._folder_info["size"] if self._folder_info["prop_attn"] else None

        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights, metric, attn_cls = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        # set metric to hidden_states itself (not key value)
        metric = hidden_states[:]
        r = self._folder_info["r"].pop(0)
        if r > (hidden_states.shape[-2]-1)//2: #fold
            r_remove = min(hidden_states.shape[-2]//2-1,r)
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
                hidden_states, self._folder_info["size"], metric, attn_cls = merge_wavg_ours(merge, hidden_states, self._folder_info["size"], metric, attn_cls)
                r_remove = min((hidden_states.shape[-2]-1)//2,r)
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
            hidden_states, self._folder_info["size"] = merge_wavg_ours(merge, hidden_states, self._folder_info["size"])
        self._folder_info["num_layer"]+=1
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


def make_folder_class(transformer_class):
    class FolderVisionTransformer(transformer_class):    
        def forward(self, *args, **kwdargs) -> torch.Tensor:
            if self.is_folder:
                self._folder_info["r"] = parse_r(self.config.num_hidden_layers, self.r)
                self._folder_info["r"][-2] += self.r #llava choose -2 layer feature, so to keep same token amount
            else:
                removed_token = self.r*self.config.num_hidden_layers
                ratio = [0,0,5]
                last_remove = int(ratio[-1]/sum(ratio)*removed_token)
                third_last_remove = int(ratio[0]/sum(ratio)*removed_token)
                self._folder_info["r"] = [0 for _ in range(self.config.num_hidden_layers-4)]+[third_last_remove, removed_token-last_remove-third_last_remove, last_remove, 0]
            self._folder_info["size"] = None
            self._folder_info["source"] = None
            self._folder_info["alpha"] = self.alpha
            self._folder_info["beta"] = self.beta
            self._folder_info["gamma"] =self.gamma
            self._folder_info["num_layer"]=self.num_layer
            self._folder_info["r_threshold"] =self.r_threshold
            return super().forward(*args, **kwdargs)
    return FolderVisionTransformer

def apply_patch(model: CLIPVisionTransformer, trace_source: bool = False, prop_attn: bool = True):
    FolderVisionTransformer = make_folder_class(model.__class__)    
    model.__class__ = FolderVisionTransformer
    model.r = 0
    model.alpha=1
    model.beta=1   #unused
    model.gamma=0  #unused
    model.num_layer=0
    model.r_threshold = 40
    model.is_folder = False #whether to use Folder or Folder
    model.cls_token = True
    model._folder_info = {"r": model.r,"size": None,"source": None,"trace_source": trace_source,"prop_attn": prop_attn,
        "class_token": model.cls_token is not None, "alpha":model.alpha,
        "beta":model.beta,"gamma":model.gamma,"num_layer":0,"r_threshold" :model.r_threshold}

    for module in model.modules():
        if isinstance(module, CLIPEncoderLayer):
            module.__class__ = FolderEncoderLayer
            module._folder_info = model._folder_info
        elif isinstance(module, CLIPAttention):
            module.__class__ = FolderAttention



