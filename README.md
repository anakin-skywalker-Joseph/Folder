# Official Implementation for FOLDER: Accelerating Multi-modal Large Language Models with Enhanced Performance
## Get Started
We here offer code implementation of FOLDER on BLIP (in **BLIP_folder**) and LLaVA1.5 (in **vlmeval_folder**). To build up the environment, please follow [BLIP](https://github.com/salesforce/BLIP) and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).
## Reproducibility
1. For BLIP, after setting up the image fold address in **BLIP_folder/configs/caption_coco_base.yaml**, go into **BLIP_folder** and run **bash run_caption.sh** to reproduce the result.
2. For LLaVA1.5 inference, please clone the repository of [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) and substitute **VLMEvalKit/vlmeval** by our **vlmeval_folder** and **run.sh** by our **run_vlmeval.sh**. Then simply run **bash run_vlmeval.sh** to reproduce the result.
## Repository Under Construction...
Tip: Folder is an improved version of Turbo (ECCV2024) for MLLM acceleration. Currently, by setting **is_turbo=True**, the approach will swith from Folder to Turbo. A complete version for both is coming soon. 
