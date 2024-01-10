"""
@file   : run_train.py
@author : Henry
@email  : luxiaonlp@163.com
@time   : 2024-01-02
"""
import os
import json
import torch
import random
import numpy as np
from model import Model
from torch import nn
from config import set_args
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from data_helper import load_data, SummaryDataset, Collator
from transformers.models.bert.tokenization_bert import BasicTokenizer
from loguru import logger
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboard import SummaryWriter

# python -m tensorboard.main --logdir=./runs --host=127.0.0.1

def set_seed(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


logger.add('./logger/log.log')

def calc_acc(logits, decoder_input_ids):
    decoder_attention_mask = torch.ne(decoder_input_ids, 0).to(logits.device)
    mask = decoder_attention_mask.view(-1).eq(1)
    labels = decoder_input_ids.view(-1)
    # print(mask.size())   # torch.Size([168])
    # print(labels.size())  # torch.Size([168])
    # print(logits.size())   # torch.Size([8, 21, 10002])
    logits = logits.contiguous().view(-1, logits.size(-1))
    # print(logits.size())   # torch.Size([168, 10002])

    _, logits = logits.max(dim=-1)
    # print(logits.size())   # torch.Size([168])
    n_correct = logits.eq(labels).masked_select(mask).sum().item()
    n_word = mask.sum().item()
    return n_correct, n_word


def evaluate(dev_data_loader):
    total_loss, total = 0.0, 0.0
    total_correct, total_word = 0.0, 0.0

    # 进行测试
    model.eval()
    for step, batch in enumerate(dev_data_loader):
        with torch.no_grad():
            if torch.cuda.is_available():
                batch = (t.cuda() for t in batch)

            context_input_ids, context_seq_len, summary_input_ids, summary_seq_len = batch
            # logits = model(context_input_ids, summary_input_ids, context_seq_len)
            logits, attention_weights, coverage_vector = model(context_input_ids, summary_input_ids, context_seq_len)
            loss = loss_func(logits, summary_input_ids, pad_id, smoothing=False)


            # 对loss进行累加
            total_loss += loss * context_input_ids.size(0)
            total += context_input_ids.size(0)

            n_correct, n_word = calc_acc(logits, summary_input_ids)
            total_correct += n_correct
            total_word += n_word
    # 计算最终测试集的loss和acc结果
    test_loss = total_loss / total
    test_acc = total_correct / total_word
    return test_loss, test_acc


def loss_func(logits, labels, pad_id, smoothing=False):
    if smoothing:
        logit = logits[..., :-1, :].contiguous().view(-1, logits.size(2))
        labels = labels[..., 1:].contiguous().view(-1)
        eps = 0.1
        n_class = logit.size(-1)
        one_hot = torch.zeros_like(logit).scatter(1, labels.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(logit, dim=1)
        non_pad_mask = labels.ne(pad_id)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()  # average later
    else:
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        labels = labels[..., 1:].contiguous().view(-1)
        loss = F.cross_entropy(logits, labels, ignore_index=pad_id)
    return loss


if __name__ == '__main__':
    args = set_args()
    set_seed(args)
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载词表
    tokenizer = BasicTokenizer()
    vocab2id = json.load(open(args.vocab2id_path, 'r', encoding='utf8'))
    pad_id = vocab2id.get('<PAD>')
    collate_fn = Collator(pad_id=pad_id, is_train=True)
    # 加载训练数据
    train_context = load_data(args.train_data_src)
    train_summary = load_data(args.train_data_tgt)
    train_dataset = SummaryDataset(context=train_context, summary=train_summary, tokenizer=tokenizer, vocab2id=vocab2id)
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=collate_fn)

    # 加载验证集
    valid_context = load_data(args.valid_data_src)
    valid_summary = load_data(args.valid_data_tgt)
    valid_dataset = SummaryDataset(context=valid_context, summary=valid_summary, tokenizer=tokenizer, vocab2id=vocab2id)
    valid_data_loader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=collate_fn)

    model = Model(vocab=vocab2id)

    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    total_steps = len(train_data_loader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.05 * total_steps,
                                                num_training_steps=total_steps)
        
    tb_write = SummaryWriter()
    global_step = 0
    tr_loss, logging_loss, min_loss = 0.0, 0.0, 0.0
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            if torch.cuda.is_available():
                batch = (t.cuda() for t in batch)
            context_input_ids, context_seq_len, summary_input_ids, summary_seq_len = batch
            logits, attention_weights, coverage_vector = model(context_input_ids, summary_input_ids, context_seq_len)

            ce_loss = loss_func(logits, summary_input_ids, pad_id, smoothing=False)
            c_t = torch.min(attention_weights, coverage_vector)
            cov_loss = torch.mean(torch.sum(c_t, dim=1))
            # 计算整体 loss
            loss = ce_loss + args.cov_lambda * cov_loss

            tr_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            # 如果步数整除logging_steps，则记录学习率和训练集损失值
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                tb_write.add_scalar("train_loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss
            # print('epoch:{}, step:{}, loss:{}'.format(epoch, step, round(loss.item(), 4)))
            logger.info('epoch:{}, step:{}, loss:{}'.format(epoch, step, round(loss.item(), 4)))

        eval_loss, eval_acc = evaluate(valid_data_loader)
        tb_write.add_scalar("test_loss", eval_loss, global_step)
        tb_write.add_scalar("test_acc", eval_acc, global_step)
        print("test_loss: {}, test_acc:{}".format(eval_loss, eval_acc))
        model.train()

        # 每个epoch进行完，则保存模型
        output_dir = os.path.join(args.output_dir, "epoch_{}.bin".format(epoch))
        torch.save(model.state_dict(), output_dir)






