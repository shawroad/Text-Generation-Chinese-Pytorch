"""
@file   : run_train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-08-03
"""
import os
import torch
import random
import numpy as np
import json
import torch.nn.functional as F
from config import set_args
from torch.optim import AdamW
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence
from data_helper import CPMDataset, load_data
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Config, CpmTokenizer
from transformers import get_linear_schedule_with_warmup


def caculate_loss(logit, target, pad_idx, smoothing=True):
    if smoothing:
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(2))
        target = target[..., 1:].contiguous().view(-1)

        eps = 0.1
        n_class = logit.size(-1)

        one_hot = torch.zeros_like(logit).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(logit, dim=1)

        non_pad_mask = target.ne(pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()  # average later
    else:
        logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
        labels = target[..., 1:].contiguous().view(-1)
        loss = F.cross_entropy(logit, labels, ignore_index=pad_idx)
    return loss


def calculate_acc(logit, labels, ignore_index):
    logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
    labels = labels[..., 1:].contiguous().view(-1)

    _, logit = logit.max(dim=-1)  # 对于每条数据，返回最大的index
    # 进行非运算，返回一个tensor，若labels的第i个位置为pad_id，则置为0，否则为1
    non_pad_mask = labels.ne(ignore_index)
    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    acc = n_correct / n_word
    return acc


def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def get_model(pretrain_model_path=None):
    # 有预训练模型  则加载  否则从头开始训练
    if pretrain_model_path is not None:
        model = GPT2LMHeadModel.from_pretrained(pretrain_model_path)  # 加载预训练模型
    else:
        model_config = GPT2Config.from_json_file(args.cpm_model_config)
        model = GPT2LMHeadModel(config=model_config)
    if torch.cuda.is_available():
        model.cuda()
    return model


def collate_fn(batch):
    input_ids = pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    return input_ids


if __name__ == '__main__':
    # 初始化参数
    args = set_args()
    # os.makedirs(args.output_dir, exist_ok=True)
    set_seed()
    # 初始化tokenizer
    tokenizer = CpmTokenizer(vocab_file=args.cpm_model_vocab)
    model = get_model()   # 这里其实还是用gpt2  只是分词采用的是CPM
    assert model.config.vocab_size == tokenizer.vocab_size

    # 计算模型参数数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('total parameters nums:', num_parameters)

    # 加载训练集 如果有预处理数据 直接加载  否则  直接预处理
    try:
        train_data_list = json.load(open(args.train_data_path_processed, 'r', encoding='utf8'))
    except:
        train_data_list = load_data(args.train_data_path, tokenizer, args.train_data_path_processed)

    train_dataset = CPMDataset(train_data_list, args.max_len)

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss, epoch_acc = 0, 0
        epoch_start_time = datetime.now()
        for step, input_ids in enumerate(train_dataloader):
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            loss = caculate_loss(logits, input_ids, tokenizer.pad_token_id, smoothing=True)
            accuracy = calculate_acc(logits, input_ids, ignore_index=tokenizer.pad_token_id)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                accuracy = accuracy / args.gradient_accumulation_steps
            loss.backward()
            print('epoch:{}, step:{}, loss:{:10f}, accuracy:{:10f}, lr:{:10f}'.format(epoch, step, loss, accuracy, scheduler.get_last_lr()[0]))

            # 梯度裁剪  一般梯度裁剪尽量别用  会影响效果
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            epoch_loss += loss.item()
            epoch_acc += accuracy

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  # 进行warm_up

        avg_loss = epoch_loss / len(train_dataloader)
        avg_acc = epoch_acc / len(train_dataloader)

        ss = 'epoch:{}, loss:{:10f}, accuracy:{:10f}'.format(epoch, avg_loss, avg_acc)
        loss_path = os.path.join(args.output_dir, 'logs.txt')
        with open(loss_path, 'a+', encoding='utf8') as f:
            f.write(ss + '\n')

        # 一个epoch跑完保存一下模型
        model_save_path = os.path.join(args.output_dir, 'model_epoch_{}'.format(epoch + 1))
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        epoch_finish_time = datetime.now()
        print('per epoch cost time: {}'.format(epoch_finish_time - epoch_start_time))
