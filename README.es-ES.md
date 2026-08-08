

# FOLDER: Aceleración de Modelos de Lenguaje Grande Multi-modal con Rendimiento Mejorado

*Este repositorio es la implementación oficial de Folder (ICCV2025) y Turbo (ECCV2024 oral)*

**FOLDER: Aceleración de Modelos de Lenguaje Grande Multi-modal con Rendimiento Mejorado** (ICCV2025) [[Paper](https://arxiv.org/pdf/2501.02430)] <br>
[Haicheng Wang*](https://scholar.google.com/citations?user=x0Uk7S8AAAAJ&hl=en), [Zhemeng Yu*](https://scholar.google.com/citations?user=1cwEkjEAAAAJ&hl=en), [Gabriele Spadaro](https://scholar.google.com/citations?hl=en&user=9uugWy0AAAAJ), [Chen Ju](https://voide1220.github.io), [Shuai Xiao](https://sites.google.com/view/xiao-shuai/home), [Victor Quétu](https://scholar.google.com/citations?hl=en&user=wfwULQUAAAAJ), [Enzo Tartaglione✉️](https://enzotarta.github.io/) (*Contribución Igual)

**Turbo: Complemento de Aceleración Impulsado por la Informatividad para Modelos de Visión-Lenguaje** (ECCV 2024, **Oral**) [[Paper](https://arxiv.org/pdf/2407.11717)] <br>
[Chen Ju*](https://voide1220.github.io), [Haicheng Wang*](https://scholar.google.com/citations?user=x0Uk7S8AAAAJ&hl=en), Haozhe Cheng, [Xu Chen](https://scholar.google.com/citations?hl=en&user=6Qa2JCwAAAAJ), [Zhonghua Zhai](https://scholar.google.com/citations?hl=en&user=o4SDCAYAAAAJ), [Weilin Huang](https://whuang.org/), Jinsong Lan, [Shuai Xiao✉️](https://sites.google.com/view/xiao-shuai/home), [Bo Zheng](https://scholar.google.com/citations?hl=en&user=3gHhO9QAAAAJ) (*Contribución Igual)

## 💡 Destacados
- 🔥 **Aceleración Universal para Diversos VLMs** Aplicable a varios tipos de VLMs, incluidos VLAs similares a CLIP, modelos de difusión y MLLMs.
- 🔥 **Mantenimiento del Rendimiento** Acelera el throughput de 1.6 a 2.0X con una caída menor en el rendimiento.
- 🔥 **Plug-and-play** Se puede aplicar directamente en la mayoría de los VLMs sin reentrenamiento. También puede utilizarse para acelerar el entrenamiento. Muy fácil de implementar (listo en 10 minutos).

## 📜 Novedades
🚀 [2025/6/26] FOLDER ha sido aceptado por ICCV2025.

🚀 [2025/2/9] Publicamos el código para BLIP y MLLMs (LLaVA1.5, Minigptv2, VITA1.5, VILA1.5, WePOINTs1.5, VideoLLaVA).

🚀 [2024/7/3] Turbo ha sido aceptado por ECCV2024 como presentación oral.

## 👨‍💻 Lista de Tareas (Todo)
- [x] Turbo para ViT
- [x] Turbo para Stable Diffusion
- [x] Puntos de control (checkpoints) de modelos reentrenados con Folder

## 🛠️ Uso

### Instalación
Para configurar el entorno, siga las instrucciones de [BLIP](https://github.com/salesforce/BLIP), [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) y los correspondientes MLLMs ([LLaVA1.5](https://github.com/haotian-liu/LLaVA), [Minigptv2](https://github.com/Vision-CAIR/MiniGPT-4), [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA), [VITA1.5](https://github.com/VITA-MLLM/VITA), [VILA1.5](https://github.com/NVlabs/VILA), [WePOINTS1.5](https://github.com/WePOINTS/WePOINTS)).

### Cómo usar

Primero, clone nuestro repositorio de GitHub ejecutando el siguiente comando.

```shell
git clone https://github.com/anakin-skywalker-Joseph/Folder.git
cd Folder
```
### Aceleración de inferencia para VLMs
La implementación de Turbo en BLIP se encuentra en [BLIP_turbo](https://github.com/anakin-skywalker-Joseph/Folder/tree/main/BLIP_turbo). Folder puede acelerar mejor BLIP en la tarea de generación de descripciones [BLIP_folder](https://github.com/anakin-skywalker-Joseph/Folder/tree/main/BLIP_folder). Por ejemplo, para BLIP_folder, después de configurar la ruta de la carpeta de imágenes en [BLIP_folder/configs/caption_coco_base.yaml](https://github.com/anakin-skywalker-Joseph/Folder/blob/main/BLIP_folder/configs/caption_coco_base.yaml), ingrese a [BLIP_folder](https://github.com/anakin-skywalker-Joseph/Folder/tree/main/BLIP_folder) y ejecute **bash run_caption.sh** para reproducir el resultado. Se puede realizar una configuración similar en [BLIP_turbo](https://github.com/anakin-skywalker-Joseph/Folder/tree/main/BLIP_turbo) para diversas tareas.

### Aceleración de inferencia para MLLMs (versión completa)
Folder es una versión mejorada de Turbo para la aceleración de MLLMs, mediante la fusión de tokens en la última capa. Proporcionamos una implementación completa para [LLaVA1.5](https://github.com/anakin-skywalker-Joseph/Folder/blob/main/llava), [Video-LLaVA](https://github.com/anakin-skywalker-Joseph/Folder/tree/main/videollava) y [Minigptv2](https://github.com/anakin-skywalker-Joseph/Folder/tree/main/minigpt4). Puede modificar la tasa de reducción en [Line 33-34](https://github.com/anakin-skywalker-Joseph/Folder/blob/main/llava/model/multimodal_encoder/clip_encoder.py#L33) para LLaVA1.5, [Line 71-72](https://github.com/anakin-skywalker-Joseph/Folder/blob/main/minigpt4/models/minigpt_v2.py#L71) para Minigptv2 y [Line 204-205](https://github.com/anakin-skywalker-Joseph/Folder/blob/main/videollava/model/multimodal_encoder/languagebind/__init__.py#L203) para Video-LLaVA. **alphavalue** es el hiperparámetro de equilibrio entre la redundancia mutua y el valor semántico en [Turbo](https://arxiv.org/pdf/2407.11717) y **rvalue** controla el número de tokens reducidos (número-de-tokens-reducidos=rvalue*num_layer, p. ej., 16 representa una tasa de reducción del 66% y 18 del 75% para LLaVA1.5). 


### Aceleración del entrenamiento para MLLMs (versión completa)
Folder también puede acelerar el entrenamiento (sirve como alternativa a pixel-shuffle/avg-pooling o término de regularización). Ofrecemos el código de entrenamiento para LLaVA1.5. Basta con reemplazar **llava** en el repositorio [LLaVA](https://github.com/haotian-liu/LLaVA) por nuestro [llava](https://github.com/anakin-skywalker-Joseph/Folder/tree/main/llava), e indicar la tasa de reducción como antes en [Line 33-34](https://github.com/anakin-skywalker-Joseph/Folder/blob/main/llava/model/multimodal_encoder/clip_encoder.py#L33).

### Aceleración de inferencia para MLLMs (versión simplificada, altamente recomendada)
Aunque la implementación de Turbo/Folder para MLLM es bastante sencilla, aún debe adaptarse a diferentes arquitecturas de codificadores de visión (y algunos modelos poseen operaciones de reducción de tokens como pooling/pixel-shuffle, lo cual puede causar problemas). Para minimizar el esfuerzo de implementación, ofrecemos una versión simplificada de Folder en [folder.py](https://github.com/anakin-skywalker-Joseph/Folder/blob/main/folder.py). Proporcionamos varios ejemplos de implementación en [folder_example](https://github.com/anakin-skywalker-Joseph/Folder/tree/main/folder_example). Basta con insertar la función [merge_features](https://github.com/anakin-skywalker-Joseph/Folder/blob/main/folder.py#L108) en cualquier lugar deseado para la reducción de tokens (p. ej., antes/después de la capa de proyección).
```python
merge_features(image_features, metric=None, size=None, r=1, class_token=True)
# image_features: (bs, seq_len, hidden_dim)
# metric: (bs, seq_len, metric_dim)  set to image_features itself if not specified
# size: default set to None
# r: number of tokens to be reduced (e.g. 300)
# class_token: whether the visual sequence contains class/cls token

```
* Recomendamos encarecidamente usar esta versión simplificada para implementación/comparación.

## Evaluación
Utilizamos [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) para realizar la evaluación. Consulte las instrucciones del repositorio y reemplace los archivos relacionados con los nuestros. Normalmente, diríjase al repositorio correspondiente y ejecute el siguiente comando para construir el entorno.
```shell
pip install -e .
```

## 👍 Agradecimientos
* [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) Fantástico kit de evaluación para MLLMs.
* [ToMe](https://github.com/facebookresearch/ToMe) Nuestro código se basa en ToMe. Gracias por este maravilloso trabajo.
* Crédito a [BLIP](https://github.com/salesforce/BLIP), [LLaVA1.5](https://github.com/haotian-liu/LLaVA), [Minigptv2](https://github.com/Vision-CAIR/MiniGPT-4), [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA), [VITA1.5](https://github.com/VITA-MLLM/VITA), [VILA1.5](https://github.com/NVlabs/VILA), [WePOINTS1.5](https://github.com/WePOINTS/WePOINTS) por sus VLMs/MLLMs de código abierto.

## Citación
Si considera que nuestro trabajo es útil para su investigación, por favor considere citarnos:
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
