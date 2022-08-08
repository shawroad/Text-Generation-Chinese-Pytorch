"""
@file   : data_helper.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-08-04
"""
import torch
import json
import pandas as pd
from torch.utils.data import Dataset


def load_data(path):
    input_text, target_text = [], []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = json.loads(line)
            input_text.append(line['title'])
            target_text.append(line['article'][:256])   # 人为截断前256个token
    df = pd.DataFrame({'input_text': input_text, 'target_text': target_text})
    return df


class MT5Dataset(Dataset):
    def __init__(self, data, tokenizer):
        self.input_text = data.input_text
        self.target_text = data.target_text
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        input_ids = self.tokenizer.encode(self.input_text[index])
        output_ids = self.tokenizer.encode(self.target_text[index])
        return {'input_ids': input_ids, 'decoder_input_ids': output_ids,
                'attention_mask': [1] * len(input_ids), 'decoder_attention_mask': [1] * len(output_ids)}


def pad_to_maxlen(input_ids, max_len, pad_value=0):
    if len(input_ids) >= max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [pad_value] * (max_len - len(input_ids))
    return input_ids


def collate_fn(batch):
    # 将输入和输出进行padding
    input_max_len = max([len(d['input_ids']) for d in batch])
    output_max_len = max([len(d['decoder_input_ids']) for d in batch])

    input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = [], [], [], []
    for item in batch:
        input_ids.append(pad_to_maxlen(item['input_ids'], max_len=input_max_len))
        attention_mask.append(pad_to_maxlen(item['attention_mask'], max_len=input_max_len))

        decoder_input_ids.append(pad_to_maxlen(item['decoder_input_ids'], max_len=output_max_len))
        decoder_attention_mask.append(pad_to_maxlen(item['decoder_attention_mask'], max_len=output_max_len))

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(attention_mask, dtype=torch.long)
    all_decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long)
    all_decoder_attention_mask = torch.tensor(decoder_attention_mask, dtype=torch.long)
    return all_input_ids, all_input_mask, all_decoder_input_ids, all_decoder_attention_mask



