"""
@file   : run_train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-08-05
"""
import torch
import time
import os
import datetime
from model import Model
from torch.optim import AdamW
from bert_model import BertConfig
from data_helper import UniLMDataset, collate_fn
from config import set_args
from tokenizer import load_bert_vocab
from torch.utils.data.dataloader import DataLoader
from transformers import get_linear_schedule_with_warmup


if __name__ == '__main__':
    args = set_args()
    # os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据集
    train_dataset = UniLMDataset(data_path=args.corpus_path, bert_vocab_path=args.bert_vocab_path)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    total_steps = int(len(train_data_loader) * args.num_train_epochs / args.gradient_accumulation_steps)

    # 实例化模型
    word2idx = load_bert_vocab(args.bert_vocab_path)
    bertconfig = BertConfig(vocab_size=len(word2idx))
    model = Model(config=bertconfig)

    # 加载预训练模型
    model.load_state_dict(torch.load(args.bert_pretrain_weight_path), strict=False)

    if torch.cuda.is_available():
        model.cuda()

    # 声明需要优化的参数 并定义相关优化器
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # 设置优化器
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)

    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss, count = 0.0, 0
        for step, batch in enumerate(train_data_loader):
            if torch.cuda.is_available():
                batch = (t.cuda() for t in batch)
            token_ids, token_type_ids, target_ids = batch

            # 因为传入了target标签，因此会计算loss并且返回
            predictions, loss = model(token_ids, token_type_ids, labels=target_ids)
            total_loss += loss.item()
            count += 1
            print('epoch:{}, step:{}, loss:{:8f}'.format(epoch, step, loss))

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        s = 'Epoch: {} | Train_AvgLoss: {:10f} '.format(epoch, total_loss / count)
        logs_path = os.path.join(args.output_dir, 'logs.txt')
        with open(logs_path, 'a+') as f:
            s += '\n'
            f.write(s)

        output_dir = os.path.join(args.output_dir, "Epoch-{}.bin".format(epoch))
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), output_dir)
        # 清空cuda缓存
        torch.cuda.empty_cache()
