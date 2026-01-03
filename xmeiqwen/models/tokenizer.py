import contextlib
import logging
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

import xmeiqwen.common.dist_utils as dist_utils
from xmeiqwen.common.dist_utils import download_cached_file
from xmeiqwen.common.utils import is_url
from xmeiqwen.common.logger import MetricLogger
from xmeiqwen.models.base_model import BaseModel
from xmeiqwen.models.Qformer import BertConfig, BertLMHeadModel
from xmeiqwen.models.eva_vit import create_eva_vit_g
from transformers import AutoTokenizer
import config


## 在 xmeiqwen 中，每个 LLM 都需要自己的 'eos', 'pad', 'bos'；否则模型会报错
def load_tokenizer_from_LLM(model_name):
    if model_name in ['Baichuan2']:
        tokenizer = AutoTokenizer.from_pretrained(config.PATH_TO_LLM[model_name], use_fast=False, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.PATH_TO_LLM[model_name], use_fast=False)
    if model_name in ['Qwen2', 'Qwen25']: tokenizer.bos_token='<|im_start|>'
    tokenizer.pad_token = tokenizer.eos_token # 看看如果全设置成这样子会有什么影响？ vicuna, llama2, llama3
    tokenizer.add_tokens([config.DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    tokenizer.add_tokens([config.DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
    tokenizer.add_tokens([config.DEFAULT_FRAME_PATCH_TOKEN], special_tokens=True)
    tokenizer.add_tokens([config.DEFAULT_FACE_PATCH_TOKEN],  special_tokens=True)
    tokenizer.add_tokens([config.DEFAULT_MULTI_PATCH_TOKEN], special_tokens=True)
    return tokenizer
