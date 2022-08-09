"""
@file   : inference.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-08-03
"""
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
    max_length = len(input_ids) + 200
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
    output_greedy = model.generate(input_ids, max_length=max_length, do_sample=False, eos_token_id=tokenizer.eos_token_id)
    res = tokenizer.decode(output_greedy[0])
    vocab = res.split('[SEP]')[1:-1]
    vocab = [v.replace(' ', '') for v in vocab]
    return vocab


def beamsearch_decode(input_ids):
    max_length = len(input_ids) + 200
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
    output_beam = model.generate(input_ids, max_length=max_length, num_beams=3, do_sample=False,
                                 no_repeat_ngram_size=2, eos_token_id=tokenizer.eos_token_id)
    res = tokenizer.decode(output_beam[0])
    vocab = res.split('[SEP]')[1:-1]
    vocab = [v.replace(' ', '') for v in vocab]
    return vocab


def greedy_sample_decode(input_ids):
    max_length = len(input_ids) + 200
    repetition_penalty = 1.1
    temperature = 1
    topk = 5
    topp = 0.95
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    output_greedy_random = model.generate(input_ids, max_length=max_length, do_sample=True,
                                          temperature=temperature, top_k=topk, top_p=topp,
                                          repetition_penalty=repetition_penalty,
                                          eos_token_id=tokenizer.eos_token_id)
    res = tokenizer.decode(output_greedy_random[0])
    vocab = res.split('[SEP]')[1:-1]
    vocab = [v.replace(' ', '') for v in vocab]
    return vocab


def beamsearch_sample_decode(input_ids):
    max_length = len(input_ids) + 200
    repetition_penalty = 1.1
    num_beams = 3
    temperature = 1
    topk = 5
    topp = 0.95
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    output_beamsearch_random = model.generate(input_ids, max_length=max_length, do_sample=True,
                                              num_beams=num_beams, temperature=temperature, top_k=topk,
                                              top_p=topp, repetition_penalty=repetition_penalty,
                                              eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    res = tokenizer.decode(output_beamsearch_random[0])
    article = res.split('[SEP]')[-1]
    article = [v.replace(' ', '') for v in article]
    return article


if __name__ == '__main__':
    args = set_args()
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)
    num_added_toks = tokenizer.add_special_tokens({'bos_token': '[BOS]', 'eos_token': '[EOS]'})

    # 加载模型
    model_path = os.path.join(args.output_dir, 'model_epoch_{}'.format(9))
    model = GPT2LMHeadModel.from_pretrained(model_path)

    if torch.cuda.is_available():
        model.cuda()
    # model.half()
    model.eval()

    title = '家乡的四季'
    prefix = '在我的家乡，春天是万物复苏的季节。'
    bos_id = tokenizer.bos_token_id   # 开始
    sep_id = tokenizer.sep_token_id   # 标题和文章分割符
    eos_id = tokenizer.eos_token_id   # 结束
    title_ids = tokenizer.encode(title, add_special_tokens=False)
    article_ids = tokenizer.encode(prefix, add_special_tokens=False)
    input_ids = [bos_id] + title_ids + [sep_id] + article_ids

    gen_article_greedy = greedy_decode(input_ids)
    gen_article_beamsearch = beamsearch_decode(input_ids)
    gen_article_greedy_sample = greedy_sample_decode(input_ids)
    gen_article_beamsearch_sample = beamsearch_sample_decode(input_ids)
    print(gen_article_greedy)
    print(gen_article_beamsearch)
    print(gen_article_greedy_sample)
    print(gen_article_beamsearch_sample)
