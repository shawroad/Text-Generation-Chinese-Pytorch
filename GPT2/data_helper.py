"""
@file   : data_helper.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-08-03
"""
import re
import torch
import json
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset


def load_data(path, tokenizer, save_path):
    # 起始和结束和分割特殊字符
    bos_id = tokenizer.bos_token_id   # 开始
    sep_id = tokenizer.sep_token_id   # 标题和文章分割符
    eos_id = tokenizer.eos_token_id   # 结束

    win_size = 200   # 窗口大小
    step_length = 128   # 步长
    train_list = []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = json.loads(line)
            title = line['title']
            article = line['article']

            title_ids = tokenizer.encode(title, add_special_tokens=False)
            article_ids = tokenizer.encode(article, add_special_tokens=False)
            token_ids = [bos_id] + title_ids + [sep_id] + article_ids + [eos_id]

            # 如果数据过长 滑动窗口处理
            start_index = 0
            end_index = win_size
            data = token_ids[start_index:end_index]
            train_list.append(data)

            start_index += step_length
            end_index += step_length
            while end_index + 50 < len(token_ids):  # 剩下的数据长度，大于或等于50，才加入训练数据集
                data = token_ids[start_index:end_index]
                train_list.append(data)
                start_index += step_length
                end_index += step_length
    json.dump(train_list, open(save_path, 'w', encoding='utf8'))
    return train_list


class GPT2Dataset(Dataset):
    def __init__(self, input_list, max_len):
        self.input_list = input_list
        self.max_len = max_len

    def __getitem__(self, index):
        input_ids = self.input_list[index]
        input_ids = input_ids[:self.max_len]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids

    def __len__(self):
        return len(self.input_list)
