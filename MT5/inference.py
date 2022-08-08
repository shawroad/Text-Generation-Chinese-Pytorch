"""
@file   : inference.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-08-04
"""
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
from tokenizer import T5PegasusTokenizer
import pandas as pd
import torch
import os
import re
import json
import requests
from config import set_args
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers import BertTokenizer
import warnings
warnings.filterwarnings("ignore")


def greedy_decode(input_ids):
    max_length = 300
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
    output_greedy = model.generate(input_ids, decoder_start_token_id=tokenizer.cls_token_id, max_length=max_length, do_sample=False, eos_token_id=tokenizer.sep_token_id)
    res = tokenizer.decode(output_greedy[0])
    res = [v.replace(' ', '') for v in res]
    res = ''.join(res)
    res = res.replace('[CLS]', '').replace('[SEP]', '')

    return res


def beamsearch_decode(input_ids):
    max_length = 300
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    output_beam = model.generate(input_ids, max_length=max_length, num_beams=3,
                                 do_sample=False,
                                 no_repeat_ngram_size=2,
                                 decoder_start_token_id=tokenizer.cls_token_id,
                                 eos_token_id=tokenizer.sep_token_id)
    res = tokenizer.decode(output_beam[0])
    res = [v.replace(' ', '') for v in res]
    res = ''.join(res)
    res = res.replace('[CLS]', '').replace('[SEP]', '')
    return res


def greedy_sample_decode(input_ids):
    max_length = 300
    repetition_penalty = 1.1
    temperature = 1
    topk = 5
    topp = 0.95
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    output_greedy_random = model.generate(input_ids, max_length=max_length, do_sample=True,
                                          temperature=temperature, top_k=topk, top_p=topp,
                                          repetition_penalty=repetition_penalty,
                                          decoder_start_token_id=tokenizer.cls_token_id,
                                          eos_token_id=tokenizer.sep_token_id)
    res = tokenizer.decode(output_greedy_random[0])
    res = [v.replace(' ', '') for v in res]
    res = ''.join(res)
    res = res.replace('[CLS]', '').replace('[SEP]', '')
    return res


def beamsearch_sample_decode(input_ids):
    max_length = 300
    repetition_penalty = 1.1
    num_beams = 3
    temperature = 1
    topk = 5
    topp = 0.95
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    output_beamsearch_random = model.generate(input_ids, max_length=max_length, do_sample=True,
                                              num_beams=num_beams, temperature=temperature, top_k=topk,
                                              top_p=topp, repetition_penalty=repetition_penalty,
                                              decoder_start_token_id=tokenizer.cls_token_id,
                                              eos_token_id=tokenizer.sep_token_id,
                                              pad_token_id=tokenizer.pad_token_id)
    res = tokenizer.decode(output_beamsearch_random[0])
    res = [v.replace(' ', '') for v in res]
    res = ''.join(res)
    res = res.replace('[CLS]', '').replace('[SEP]', '')
    return res


if __name__ == '__main__':
    args = set_args()
    checkpoint = '/usr/home/xiaolu10/xiaolu10/gpu_task/text_generation/MT5/checkpoint/model_epoch_10'
    tokenizer = T5PegasusTokenizer.from_pretrained(checkpoint)
    model = MT5ForConditionalGeneration.from_pretrained(checkpoint)

    if torch.cuda.is_available():
        model.cuda()
    # model.half()
    model.eval()

    title = '我的祖国'
    title_ids = tokenizer.encode(title)

    gen_article_greedy = greedy_decode(title_ids)
    gen_article_beamsearch = beamsearch_decode(title_ids)
    gen_article_greedy_sample = greedy_sample_decode(title_ids)
    gen_article_beamsearch_sample = beamsearch_sample_decode(title_ids)
    print(gen_article_greedy)
    print(gen_article_beamsearch)
    print(gen_article_greedy_sample)
    print(gen_article_beamsearch_sample)
