# Enhancing User Behavior Sequence Modeling by Generative Tasks for Session Search

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)

## Abstract
This repository contains the source code for the CIKM paper [Enhancing User Behavior Sequence Modeling by Generative Tasks for Session Search](https://dl.acm.org/doi/10.1145/3511808.3557310) by Chen et al. <br>

Users' search tasks have become increasingly complicated, requiring multiple queries and interactions with the results. Recent studies have demonstrated that modeling the historical user behaviors in a session can help understand the current search intent. Existing context-aware ranking models primarily encode the current session sequence (from the first behavior to the current query) and compute the ranking score using the high-level representations. However, there is usually some noise in the current session sequence (useless behaviors for inferring the search intent) that may affect the quality of the encoded representations. To help the encoding of the current user behavior sequence, we propose to use a decoder and the information of future sequences and a supplemental query. Specifically, we design three generative tasks that can help the encoder to infer the actual search intent: (1) predicting future queries, (2) predicting future clicked documents, and (3) predicting a supplemental query. We jointly learn the ranking task with these generative tasks using an encoder-decoder structured approach. Extensive experiments on two public search logs demonstrate that our model outperforms all existing baselines, and the designed generative tasks can actually help the ranking task. Besides, additional experiments also show that our approach can be easily applied to various Transformer-based encoder-decoder models and improve their performance.

Authors: Haonan Chen, Zhicheng Dou, Yutao Zhu, Zhao Cao, Xiaohua Cheng, and Ji-rong Wen


## Requirements
- Python 3.7.13 <br>
- Pytorch 1.11.0 (with GPU support) <br>
- Transformers 4.18.0 <br>
- [pytrec-eval](https://pypi.org/project/pytrec-eval/) 0.5  

## Usage
- data 
  - AOL: Please reach to the author of [CARS](https://arxiv.org/pdf/1906.02329.pdf)
  - Tiangong-ST: [Download here](http://www.thuir.cn/tiangong-st/)
- Prepare pretrained BART
  - [BARTBase](https://huggingface.co/facebook/bart-base)
  - [BartChinese](https://huggingface.co/fnlp/bart-base-chinese)  
  - Save to the "pretrained_model" directory

## Run
- For Training:
```
python runASE.py --dataset aol --do_train True --do_eval True
```
- For Testing:
```
python runASE.py --dataset aol --do_train False --do_eval True
```

## Citations
If you use the code, please cite the following paper:  
```
@inproceedings{CDZCCW2022CIKM,
author = {Chen, Haonan and Dou, Zhicheng and Zhu, Yutao and Cao, Zhao and Cheng, Xiaohua and Wen, Ji-Rong},
title = {Enhancing User Behavior Sequence Modeling by Generative Tasks for Session Search},
year = {2022},
url = {https://doi.org/10.1145/3511808.3557310},
doi = {10.1145/3511808.3557310},
pages = {180–190},
}
```
