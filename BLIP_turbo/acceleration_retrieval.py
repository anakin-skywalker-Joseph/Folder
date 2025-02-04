import argparse
import os
import ruamel.yaml as yaml
# import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from models.med import BertConfig, BertModel, BertLMHeadModel
from models.blip import create_vit, init_tokenizer, load_checkpoint
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.vqa_dataset import vqa_collate_fn
from data.utils import save_result
import ipdb
from utils import print_params_and_flops
import turbo
from models.vit_retrieval import VisionTransformer
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

def create_vit_retrieval(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0,r_value=40,alpha=1):
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit=='base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12, 
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                          )

    elif vit=='large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24, 
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                          ) 

    return visual_encoder, vision_width

class BLIP_IMAGE(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0                 
                 ):           
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit_retrieval(vit, image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1)
        self.tokenizer = init_tokenizer()  
        
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)   #text encoder
        # turbo.patch.bertmm(self.text_encoder)
        decoder_config = BertConfig.from_json_file(med_config)        
        self.text_decoder = BertLMHeadModel(config=decoder_config)
    def return_encoder(self):
        return self.visual_encoder

    def forward(self, image):
        image_embeds = self.visual_encoder(image)   #visual encoder
        return image_embeds

def blip_image(pretrained='',**kwargs):
    model = BLIP_IMAGE(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
#         assert(len(msg.missing_keys)==0)
    return model  

@torch.no_grad()
def main(args, config, r_value=40, alpha=1):   
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating model")
    model1 = blip_image(pretrained=config['pretrained'], image_size=config['image_size'], 
                       vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
    # ipdb.set_trace()
    model=model1.return_encoder()
    if r_value!=0:  
        if alpha!=0:
            turbo.patch.re_turbo(model)
            model.r=r_value
            model.alpha=alpha
            model.r_threshold=100
        else:
            turbo.patch.re_turbo_tome(model)
            model.r=r_value
            model.r_threshold=100
    is_cuda = args.device=="cuda" 
    input_size=(3,config["image_size"],config["image_size"])
    model = model.eval().to(device)   
    batch_size = 180
    runs=70
    warm_up=9
    total=0
    input = torch.rand(batch_size, *input_size, device=device)   
    with torch.autocast(device.type):
        with torch.no_grad():   
            for i in tqdm(range(runs), desc="Benchmarking"):
                if i == warm_up:
                    if is_cuda:
                        torch.cuda.synchronize()
                    total = 0
                    start = time.time()
                a=model(input)
                total += batch_size  
    if is_cuda:
        torch.cuda.synchronize()

    end = time.time()
    elapsed = end - start   

    throughput = total / elapsed   ## 吞吐

    print(f"r: {r_value}, Throughput: {throughput:.2f} im/s")
    with open("accelerate_retrieval_coco.txt","a") as ac:
        ac.write(f"r: {r_value}, Throughput: {throughput:.2f} im/s\n")
    flops=print_params_and_flops("retrieval",model,device)
    with open("accelerate_retrieval_coco.txt","a") as ac:
        ac.write(f"r: {r_value}, flops:{flops}\n")    
    return throughput                 
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/retrieval_flickr.yaml')
    #parser.add_argument('--config', default='./configs/retrieval_coco.yaml')       
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=40, type=int)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    r_val=[40]
    alpha_val=[0,1]
    for r in r_val:
        for a in alpha_val:
            main(args, config, r, a)
            time.sleep(2)