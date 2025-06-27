# FOLDER: Accelerating Multi-modal Large Language Models with Enhanced Performance

*This repository is the official implementation of Folder (ICCV2025) and Turbo (ECCV2024 oral)*

**FOLDER: Accelerating Multi-modal Large Language Models with Enhanced Performance** (ICCV2025) [[Paper](https://arxiv.org/pdf/2501.02430)] <br>
[Haicheng Wang*](https://scholar.google.com/citations?user=x0Uk7S8AAAAJ&hl=en), [Zhemeng Yu*](https://scholar.google.com/citations?user=1cwEkjEAAAAJ&hl=en), [Gabriele Spadaro](https://scholar.google.com/citations?hl=en&user=9uugWy0AAAAJ), [Chen Ju](https://voide1220.github.io), [Shuai Xiao](https://sites.google.com/view/xiao-shuai/home), [Victor Quétu](https://scholar.google.com/citations?hl=en&user=wfwULQUAAAAJ), [Enzo Tartaglione✉️](https://enzotarta.github.io/) (*Equal Contribution)

**Turbo: Informativity-Driven Acceleration Plug-In for Vision-Language Models** (ECCV 2024, **Oral**) [[Paper](https://arxiv.org/pdf/2407.11717)] <br>
[Chen Ju*](https://voide1220.github.io), [Haicheng Wang*](https://scholar.google.com/citations?user=x0Uk7S8AAAAJ&hl=en), Haozhe Cheng, [Xu Chen](https://scholar.google.com/citations?hl=en&user=6Qa2JCwAAAAJ), [Zhonghua Zhai](https://scholar.google.com/citations?hl=en&user=o4SDCAYAAAAJ), [Weilin Huang](https://whuang.org/), Jinsong Lan, [Shuai Xiao✉️](https://sites.google.com/view/xiao-shuai/home), [Bo Zheng](https://scholar.google.com/citations?hl=en&user=3gHhO9QAAAAJ) (*Equal Contribution)

## 💡 Highlights
- 🔥 **Universal Acceleration for Various VLMs** Applicable on various types of VLMs, including CLIP-like VLAs, diffusions and MLLMs.
- 🔥 **Performace Maintenance** Accelerate throughput 1.6-2.0X with minor performance drop.
- 🔥 **Plug-and-play** Can be directly applied in most of VLMs without retraining. Can also be used for training acceleration. Very easy to implement (10-min-ready).

## 📜 News
🚀 [2025/6/26] FOLDER has been accepted by ICCV2025.

🚀 [2025/2/9] We release the code for BLIP and MLLMs (LLaVA1.5, Minigptv2, VITA1.5, VILA1.5, WePOINTs1.5, VideoLLaVA).

🚀 [2024/7/3] Turbo has been accepted by ECCV2024 as oral presentation.

## 👨‍💻 Todo
- [x] Turbo for ViT
- [x] Turbo for Stable Diffusion
- [x] Checkpoints of Folder retrained models

## 🛠️ Usage

### Installation
To build up the environment, please follow [BLIP](https://github.com/salesforce/BLIP), [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) and corresponding MLLMs ([LLaVA1.5](https://github.com/haotian-liu/LLaVA), [Minigptv2](https://github.com/Vision-CAIR/MiniGPT-4), [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA), [VITA1.5](https://github.com/VITA-MLLM/VITA), [VILA1.5](https://github.com/NVlabs/VILA), [WePOINTS1.5](https://github.com/WePOINTS/WePOINTS)).

### How to use

Please first clone our repo from github by running the following command.

```shell
git clone https://github.com/anakin-skywalker-Joseph/Folder.git
cd Folder
```
### VLMs inference acceleration
The implementation of Turbo on BLIP is in [BLIP_turbo](https://github.com/anakin-skywalker-Joseph/Folder/tree/main/BLIP_turbo). Folder can better accelerate BLIP on captioning task [BLIP_folder](https://github.com/anakin-skywalker-Joseph/Folder/tree/main/BLIP_folder). For example, for BLIP_folder, after setting up the image fold address in [BLIP_folder/configs/caption_coco_base.yaml](https://github.com/anakin-skywalker-Joseph/Folder/blob/main/BLIP_folder/configs/caption_coco_base.yaml), go into [BLIP_folder](https://github.com/anakin-skywalker-Joseph/Folder/tree/main/BLIP_folder) and run **bash run_caption.sh** to reproduce the result. Similar setup can be done in [BLIP_turbo](https://github.com/anakin-skywalker-Joseph/Folder/tree/main/BLIP_turbo) for various tasks.

### MLLMs inference acceleration (complete version)
Folder is an upgraded version of Turbo for MLLMs acceleration, by merging tokens in the last layer. We provide complete implementation for [LLaVA1.5](https://github.com/anakin-skywalker-Joseph/Folder/blob/main/llava), [Video-LLaVA](https://github.com/anakin-skywalker-Joseph/Folder/tree/main/videollava) and [Minigptv2](https://github.com/anakin-skywalker-Joseph/Folder/tree/main/minigpt4). You can modify the reduction ratio in [Line 33-34](https://github.com/anakin-skywalker-Joseph/Folder/blob/main/llava/model/multimodal_encoder/clip_encoder.py#L33) for LLaVA1.5, [Line 71-72](https://github.com/anakin-skywalker-Joseph/Folder/blob/main/minigpt4/models/minigpt_v2.py#L71) for Minigptv2 and [Line 204-205](https://github.com/anakin-skywalker-Joseph/Folder/blob/main/videollava/model/multimodal_encoder/languagebind/__init__.py#L203) for Video-LLaVA. **alphavalue** is the balancing hyperparameter between mutual redundancy and semantic value in [Turbo](https://arxiv.org/pdf/2407.11717) and **rvalue** controls the number of tokens reduced (number-of-reduced-token=rvalue*num_layer, e.g. 16 is 66% reduction ratio and 18 is 75% reduction ratio for LLaVA1.5). 


### MLLMs training acceleration (complete version)
Folder can also accelerate the training (serves as an alternative for pixel-shuffle/avg-pooling or regularization term). We offer training code for LLaVA1.5. It's sufficient to replace **llava** in [LLaVA](https://github.com/haotian-liu/LLaVA) repo by our [llava](https://github.com/anakin-skywalker-Joseph/Folder/tree/main/llava), and indicate the reduction ratio as before in [Line 33-34](https://github.com/anakin-skywalker-Joseph/Folder/blob/main/llava/model/multimodal_encoder/clip_encoder.py#L33).

### MLLMs inference acceleration (simplified version, strongly recommand)
Although the implementation of Turbo/Folder for MLLM is rather simple, it still needs to adapt for different vision encoder architectures (and some models possess token reduction operation like pooling/pixel-shuffle, which may cause problems). In order to minimize the deployment effort, we offer a simplied version of Folder in [folder.py](https://github.com/anakin-skywalker-Joseph/Folder/blob/main/folder.py). We provide several implementation examples in [folder_example](https://github.com/anakin-skywalker-Joseph/Folder/tree/main/folder_example). It's sufficient to insert the function [merge_features](https://github.com/anakin-skywalker-Joseph/Folder/blob/main/folder.py#L108) into any desired place for token reduction (e.g. before/after projection layer).
```python
merge_features(image_features, metric=None, size=None, r=1, class_token=True)
# image_features: (bs, seq_len, hidden_dim)
# metric: (bs, seq_len, metric_dim)  set to image_features itself if not specified
# size: default set to None
# r: number of tokens to be reduced (e.g. 300)
# class_token: whether the visual sequence contains class/cls token

```
* We strongly recommand using this simplified version for deployment/comparison.

## Evaluation
We leverage [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to do the evaluation. Please refer to the repo instruction and replace the related files with ours. Normally, by going to the corresponding repo and run the following command to build the environment.
```shell
pip install -e .
```

## 👍 Acknowledgement
* [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) Fantastic MLLMs evaluation toolkit.
* [ToMe](https://github.com/facebookresearch/ToMe) Our code is based on ToMe. Thanks for this wonderful work.
* Credit on [BLIP](https://github.com/salesforce/BLIP), [LLaVA1.5](https://github.com/haotian-liu/LLaVA), [Minigptv2](https://github.com/Vision-CAIR/MiniGPT-4), [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA), [VITA1.5](https://github.com/VITA-MLLM/VITA), [VILA1.5](https://github.com/NVlabs/VILA), [WePOINTS1.5](https://github.com/WePOINTS/WePOINTS) for their open-source VLMs/MLLMs.

## Citation
If you find our work helpful for your research, please consider citing:
```
@inproceedings{ju2024turbo,
  title={Turbo: Informativity-driven acceleration plug-in for vision-language large models},
  author={Ju, Chen and Wang, Haicheng and Cheng, Haozhe and Chen, Xu and Zhai, Zhonghua and Huang, Weilin and Lan, Jinsong and Xiao, Shuai and Zheng, Bo},
  booktitle={European Conference on Computer Vision},
  pages={436--455},
  year={2024},
  organization={Springer}
}

@article{wang2025folder,
  title={FOLDER: Accelerating Multi-modal Large Language Models with Enhanced Performance},
  author={Wang, Haicheng and Yu, Zhemeng and Spadaro, Gabriele and Ju, Chen and Qu{\'e}tu, Victor and Tartaglione, Enzo},
  journal={arXiv preprint arXiv:2501.02430},
  year={2025}
}
```
