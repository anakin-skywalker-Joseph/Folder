## Turbo: Informativity-Driven Acceleration Plug-In for Vision-Language Models

In this respository, we offer a sample code of turbo module plugged in BLIP for various tasks, you may download the datasets and checkpoints listed below to test the effectiveness of our method. After preparing the environment (please refer to BLIP environment), You can run directly the script file of each task or edit the key parameters (drop ratio, balancing coefficient, distributed inference and so on).

### Pre-trained checkpoints:
Num. pre-train images | BLIP w/ ViT-B | BLIP w/ ViT-B and CapFilt-L | BLIP w/ ViT-L 
--- | :---: | :---: | :---: 
14M | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth">Download</a>| - | -
129M | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth">Download</a>| <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth">Download</a> | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth">Download</a>

### Finetuned checkpoints:
Task | BLIP w/ ViT-B | BLIP w/ ViT-B and CapFilt-L | BLIP w/ ViT-L 
--- | :---: | :---: | :---:
Image-Text Retrieval (COCO) | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth">Download</a>| - | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth">Download</a>
Image-Text Retrieval (Flickr30k) | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_flickr.pth">Download</a>|  - | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_flickr.pth">Download</a>
Image Captioning (COCO) | - | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth">Download</a>| <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth">Download</a> | 
VQA | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_vqa.pth">Download</a>| <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth">Download</a> | - 
NLVR2 | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_nlvr.pth">Download</a>| - | - 


### Image-Text Retrieval:
1. Download COCO and Flickr30k datasets from the original websites, and set 'image_root' in configs/retrieval_{dataset}.yaml accordingly.
3. To use the pre-trained checkpoint, set 'pretrained' in configs/retrieval_coco.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth". 

### Image-Text Captioning:
1. Download COCO datasets from the original websites, and set 'image_root' in configs/caption_coco.yaml.
4. To use the pre-trained checkpoint, set 'pretrained' in configs/caption_coco.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth". 

### VQA:
1. Download VQA v2 dataset and Visual Genome dataset from the original websites, and set 'vqa_root' and 'vg_root' in configs/vqa.yaml.
3. To use the pre-trained checkpoint, set 'pretrained' in configs/vqa.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth".

### NLVR2:
1. Download NLVR2 dataset from the original websites, and set 'image_root' in configs/nlvr.yaml.
3. To use the pre-trained checkpoint, set 'pretrained' in configs/nlvr.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth".  

## Run Code
Here are some examples of how we run the code:
On VQA: <pre> python -m torch.distributed.run --nproc_per_node=8 test_vqa.py --evaluate --r_value 40 --alpha 5 --beta 1 --gamma 0<\pre>
On retrieval-coco: python -m torch.distributed.run --nproc_per_node=8 test_retrieval.py --evaluate --config './configs/retrieval_coco.yaml' --output_dir 'output/Retrieval_coco' --r_value 30 --alpha 5 --beta 1 --gamma 0
--r_value is the drop ratio and --alpha is balancing coefficient. You could also adjust the merging strategy by editing --beta and --gamma or add threshold by --threshold.

We could test the acceleration performance by simply running the acceleration_xxx.py for different tasks. You could change the drop ratio inside the file to test the relationship between drop ratio and acceleration speed-up.