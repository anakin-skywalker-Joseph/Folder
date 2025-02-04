import argparse
import os
import ruamel.yaml as yaml

import numpy as np
import random
import time
import datetime
import json
from utils import print_params_and_flops
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from models.nlvr_encoder import BertModel
from models.blip import create_vit, init_tokenizer, is_url
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.vqa_dataset import vqa_collate_fn
from data.utils import save_result
import ipdb
import turbo
from timm.models.hub import download_cached_file
from models.vit import interpolate_pos_embed
from models.med import BertConfig
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

class BLIP_NLVR(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 480,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                   
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False) 
                    
        self.cls_head = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, 2)
                )  

    def forward(self, image, text):
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        image0_embeds, image1_embeds = torch.split(image_embeds,len(text))     
        text = self.tokenizer(text, padding='longest', return_tensors="pt").to(image.device) 
        text.input_ids[:,0] = self.tokenizer.enc_token_id        
        output = self.text_encoder(text.input_ids, 
                                   attention_mask = text.attention_mask, 
                                   encoder_hidden_states = [image0_embeds,image1_embeds],
                                   encoder_attention_mask = [image_atts[:image0_embeds.size(0)],
                                                             image_atts[image0_embeds.size(0):]],        
                                   return_dict = True,
                                  )  
        hidden_state = output.last_hidden_state[:,0,:]        
        prediction = self.cls_head(hidden_state)
        return prediction
    
def blip_nlvr(pretrained='',**kwargs):
    model = BLIP_NLVR(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        print("missing keys:")
        print(msg.missing_keys)
    return model  

        
def load_checkpoint(model,url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
    state_dict = checkpoint['model']
    
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
    
    for key in list(state_dict.keys()):
        if 'crossattention.self.' in key:
            new_key0 = key.replace('self','self0')
            new_key1 = key.replace('self','self1')
            state_dict[new_key0] = state_dict[key]
            state_dict[new_key1] = state_dict[key]
        elif 'crossattention.output.dense.' in key:
            new_key0 = key.replace('dense','dense0')
            new_key1 = key.replace('dense','dense1')
            state_dict[new_key0] = state_dict[key]
            state_dict[new_key1] = state_dict[key]  
                
    msg = model.load_state_dict(state_dict,strict=False)
    print('load checkpoint from %s'%url_or_filename)  
    return model,msg

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
    model1 = blip_nlvr(pretrained=config['pretrained'], image_size=config['image_size'], 
                       vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])

    if r_value!=0:  
        if alpha!=0:
            turbo.patch.turbo(model1.visual_encoder)
            model1.visual_encoder.r=r_value
            model1.visual_encoder.alpha=alpha
        else:
            turbo.patch.turbo_tome(model1.visual_encoder)
            model1.visual_encoder.r=r_value
    is_cuda = args.device=="cuda" 
    input_size=(3,config["image_size"],config["image_size"])
    text1="Are these two pictures shown here related to each other?"
    model1 = model1.eval().to(device)   
    batch_size = 100
    runs=70
    warm_up=9
    total=0
    input = torch.rand(batch_size*2, *input_size, device=device)  
    input_text = [text1 for _ in range(batch_size)] 
    with torch.autocast(device.type):
        with torch.no_grad():   
            for i in tqdm(range(runs), desc="Benchmarking"):
                if i == warm_up:
                    if is_cuda:
                        torch.cuda.synchronize()
                    total = 0
                    start = time.time()
                a=model1(input,input_text)
                total += batch_size  
    if is_cuda:
        torch.cuda.synchronize()

    end = time.time()
    elapsed = end - start   

    throughput = total / elapsed   ## 吞吐

    print(f"r: {r_value}, Throughput: {throughput:.2f} im/s")
    with open("accelerate_nlvr.txt","a") as ac:
        ac.write(f"r: {r_value}, Throughput: {throughput:.2f} im/s\n")
    flops=print_params_and_flops("nlvr",model1,device)
    with open("accelerate_nlvr.txt","a") as ac:
        ac.write(f"r: {r_value}, flops:{flops}\n") 
    return throughput                 
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/nlvr.yaml')       
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=40, type=int)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    r_val=[0,35,45]
    alpha_val=[0,1]
    for r in r_val:
        for a in alpha_val:
            main(args, config, r, a)
            time.sleep(2)