"""
@file   : data_helper.py
@author : Henry
@email  : luxiaonlp@163.com
@time   : 2024-01-06
"""
import torch
from torch.utils.data import Dataset


def load_data(path):
    data = []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data.append(line)
    return data


class SummaryDataset(Dataset):
    def __init__(self, context, summary, tokenizer, vocab2id):
        self.context = context
        self.summary = summary
        self.tokenizer = tokenizer
        self.vocab2id = vocab2id

    def __len__(self):
        return len(self.context)

    def __getitem__(self, item):
        # 输入
        context = self.context[item]
        # context_token = self.tokenizer.tokenize(context)
        context_token = list(context)
        context_seq_len = len(context_token)
        context_input_ids = []
        for word in context_token:
            context_input_ids.append(self.vocab2id.get(word, self.vocab2id.get('<UNK>')))

        # 输出
        summary = self.summary[item]
        
        summary_token = list(summary)
        summary_seq_len = len(summary_token)
        summary_input_ids = []
        
        summary_input_ids.append(self.vocab2id.get('<SOS>'))
        for word in summary_token:
            summary_input_ids.append(self.vocab2id.get(word, self.vocab2id.get('<UNK>')))
        summary_input_ids.append(self.vocab2id.get('<EOS>'))
        return {'context_input_ids': context_input_ids, 'context_seq_len': context_seq_len,
                'summary_input_ids': summary_input_ids, 'summary_seq_len': summary_seq_len}


class Collator:
    def __init__(self, pad_id, is_train=True):
        self.pad_id = pad_id
        self.is_train = is_train

    def __call__(self, batch):
        context_max_len = max([d['context_seq_len'] for d in batch])
        summary_max_len = max([d['summary_seq_len'] for d in batch])

        if context_max_len > 256:
            context_max_len = 256
        if summary_max_len > 64:
            summary_max_len = 64

        context_input_ids_list, context_seq_len_list = [], []
        summary_input_ids_list, summary_seq_len_list = [], []
        for item in batch:
            context_input_ids_list.append(self.pad_to_maxlen(item['context_input_ids'], max_len=context_max_len))
            summary_input_ids_list.append(self.pad_to_maxlen(item['summary_input_ids'], max_len=summary_max_len))
            context_seq_len_list.append(item['context_seq_len'])
            summary_seq_len_list.append(item['summary_seq_len'])

        context_input_ids_tensor = torch.tensor(context_input_ids_list, dtype=torch.long)
        summary_input_ids_tensor = torch.tensor(summary_input_ids_list, dtype=torch.long)
        context_seq_len_tensor = torch.tensor(context_seq_len_list, dtype=torch.long)
        summary_seq_len_tensor = torch.tensor(summary_seq_len_list, dtype=torch.long)
        return context_input_ids_tensor, context_seq_len_tensor, summary_input_ids_tensor, summary_seq_len_tensor

    def pad_to_maxlen(self, input_ids, max_len):
        if len(input_ids) >= max_len:
            input_ids = input_ids[:max_len]
        else:
            input_ids = input_ids + [self.pad_id] * (max_len - len(input_ids))
        return input_ids
