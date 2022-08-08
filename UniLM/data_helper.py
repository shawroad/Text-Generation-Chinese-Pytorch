"""
@file   : data_helper.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-08-05
"""
import torch
from torch.utils.data.dataset import Dataset
from tokenizer import load_bert_vocab, Tokenizer
import json


def load_data(data_path):
    sents_src = []
    sents_tgt = []
    with open(data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = json.loads(line)
            title = line['title']
            article = line['article'][:256]   # 直接截断到256
            sents_src.append(title)
            sents_tgt.append(article)
    return sents_src, sents_tgt


class UniLMDataset(Dataset):
    def __init__(self, data_path, bert_vocab_path):
        super(UniLMDataset, self).__init__()
        self.sents_src, self.sents_tgt = load_data(data_path)
        self.word2idx = load_bert_vocab(bert_vocab_path)
        self.idx2word = {k: v for v, k in self.word2idx.items()}
        self.tokenizer = Tokenizer(self.word2idx)

    def __getitem__(self, i):
        # 得到单个数据
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]

        token_ids, token_type_ids = self.tokenizer.encode(src, tgt)
        output = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
        }
        return output

    def __len__(self):
        return len(self.sents_src)


def padding(indice, max_length, pad_idx=0):
    temp = []
    for item in indice:
        if len(item) >= max_length:
            item = item[:max_length]
            temp.append(item)
        else:
            item = item + [pad_idx] * (max_length - len(item))
            temp.append(item)
    return torch.tensor(temp)


def collate_fn(batch):
    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])   # 计算当前batch的最大长度
    if max_length > 300:
        max_length = 300
    token_type_ids = [data["token_type_ids"] for data in batch]
    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()
    return token_ids_padded, token_type_ids_padded, target_ids_padded
