"""
@file   : run_train.py
@author : Henry
@email  : luxiaonlp@163.com
@time   : 2024-01-06
"""
import os
import json
import numpy as np
from torch import nn
import torch.utils.data
from config import set_args
from model import Transformer
from loguru import logger
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BasicTokenizer
from data_helper import SummaryDataset, load_data, Collator
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboard import SummaryWriter

logger.add('./logger/log.log')



def create_masks(question, reply_input, reply_target):
    def subsequent_mask(size):
        mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        return mask.unsqueeze(0)
    question_mask = (question!=0)
    question_mask = question_mask.unsqueeze(1).unsqueeze(1)         # (batch_size, 1, 1, max_words)
     
    reply_input_mask = reply_input!=0
    reply_input_mask = reply_input_mask.unsqueeze(1)  # (batch_size, 1, max_words)
    reply_input_mask = reply_input_mask & subsequent_mask(reply_input.size(-1)).type_as(reply_input_mask.data) 
    reply_input_mask = reply_input_mask.unsqueeze(1) # (batch_size, 1, max_words, max_words)
    reply_target_mask = reply_target!=0              # (batch_size, max_words)
    
    return question_mask, reply_input_mask, reply_target_mask
    

class AdamWarmup:
    def __init__(self, model_size, warmup_steps, optimizer):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.current_step = 0
        self.lr = 0

    def get_lr(self):
        return self.model_size ** (-0.5) * min(self.current_step ** (-0.5),
                                               self.current_step * self.warmup_steps ** (-1.5))

    def step(self):
        # Increment the number of steps each time we call the step function
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        # update the learning rate
        self.lr = lr
        self.optimizer.step()


class LossWithLS(nn.Module):
    def __init__(self, size, smooth):
        super(LossWithLS, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        self.confidence = 1.0 - smooth
        self.smooth = smooth
        self.size = size

    def forward(self, prediction, target, mask):
        """
        prediction of shape: (batch_size, max_words, vocab_size)
        target and mask of shape: (batch_size, max_words)
        """
        # prediction: batch_size, max_len, vocab_size
        prediction = prediction.view(-1, prediction.size(-1))  # (batch_size * max_words, vocab_size)
        # batch_size*max_len, vocab_size
        # batch_size*max_len
        target = target.contiguous().view(-1)  # (batch_size * max_words)
        mask = mask.float()
        mask = mask.view(-1)  # (batch_size * max_words)
        labels = prediction.data.clone()

        # 平滑
        labels.fill_(self.smooth / (self.size - 1))
        labels.scatter_(1, target.data.unsqueeze(1), self.confidence)
        loss = self.criterion(prediction, labels)  # (batch_size * max_words, vocab_size)
        loss = (loss.sum(1) * mask).sum() / mask.sum()
        return loss


def calc_acc(logits, decoder_input_ids):
    decoder_attention_mask = torch.ne(decoder_input_ids, 0).to(logits.device)
    mask = decoder_attention_mask.view(-1).eq(1)
    labels = decoder_input_ids.reshape(-1)
    # labels = decoder_input_ids.view(-1)
    logits = logits.contiguous().view(-1, logits.size(-1))

    _, logits = logits.max(dim=-1)
    n_correct = logits.eq(labels).masked_select(mask).sum().item()
    n_word = mask.sum().item()
    return n_correct, n_word


def evaluate(valid_data_loader):
    model.eval()
    total_loss, total = 0.0, 0.0
    total_correct, total_word = 0.0, 0.0
    # 进行测试
    model.eval()
    for step, batch in enumerate(valid_data_loader):
        with torch.no_grad():
            if torch.cuda.is_available():
                batch = (t.cuda() for t in batch)

            s_input_ids, context_seq_len, t_input_ids, summary_seq_len = batch
            t_input_ids_input = t_input_ids[:, :-1]  # [SOS ...]
            t_input_ids_output = t_input_ids[:, 1:]  # [...  EOS]

            # 自己看
            s_input_ids_mask, t_input_ids_input_mask, t_input_ids_output_mask = create_masks(s_input_ids,
                                                                                             t_input_ids_input,
                                                                                             t_input_ids_output)
            if torch.cuda.is_available():
                s_input_ids_mask, t_input_ids_input_mask, t_input_ids_output_mask = s_input_ids_mask.cuda(), t_input_ids_input_mask.cuda(), t_input_ids_output_mask.cuda()
            output = model(s_input_ids, s_input_ids_mask, t_input_ids_input, t_input_ids_input_mask)
            loss = loss_func(output, t_input_ids_output, t_input_ids_output_mask)
            # 对loss进行累加
            total_loss += loss.item() * s_input_ids.size(0)
            total += s_input_ids.size(0)

            n_correct, n_word = calc_acc(output, t_input_ids_output)
            total_correct += n_correct
            total_word += n_word
    # 计算最终测试集的loss和acc结果
    test_loss = total_loss / total
    test_acc = total_correct / total_word
    return test_loss, test_acc


if __name__ == '__main__':
    args = set_args()
    os.makedirs(args.output_dir, exist_ok=True)
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

    model = Transformer(d_model=args.d_model, heads=args.heads, num_layers=args.num_layers, word_map=vocab2id)
    if torch.cuda.is_available():
        model.cuda()

    # 优化
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    transformer_optimizer = AdamWarmup(model_size=args.d_model, warmup_steps=4000, optimizer=adam_optimizer)

    # 损失函数
    loss_func = LossWithLS(len(vocab2id), 0.1)
    tb_write = SummaryWriter()
    global_step = 0
    tr_loss, logging_loss, min_loss = 0.0, 0.0, 0.0
    for epoch in range(args.epochs):
        loss_lists = []
        for step, batch in enumerate(train_data_loader):
            if torch.cuda.is_available():
                batch = (t.cuda() for t in batch)
            s_input_ids, context_seq_len, t_input_ids, summary_seq_len = batch
            t_input_ids_input = t_input_ids[:, :-1]  # [SOS ...]
            t_input_ids_output = t_input_ids[:, 1:]  # [...  EOS]

            # 自己看
            s_input_ids_mask, t_input_ids_input_mask, t_input_ids_output_mask = create_masks(s_input_ids, t_input_ids_input, t_input_ids_output)
            
            if torch.cuda.is_available():
                s_input_ids_mask, t_input_ids_input_mask, t_input_ids_output_mask = s_input_ids_mask.cuda(), t_input_ids_input_mask.cuda(), t_input_ids_output_mask.cuda()
                
            output = model(s_input_ids, s_input_ids_mask, t_input_ids_input, t_input_ids_input_mask)
            loss = loss_func(output, t_input_ids_output, t_input_ids_output_mask)

            tr_loss += loss.item()

            transformer_optimizer.optimizer.zero_grad()
            loss.backward()
            transformer_optimizer.step()

            global_step += 1
            # 如果步数整除logging_steps，则记录学习率和训练集损失值
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                tb_write.add_scalar("train_loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss
            loss = loss.item()
            logger.info('epoch:{}, step:{}, loss:{}'.format(epoch, step, round(loss, 4)))

        eval_loss, eval_acc = evaluate(valid_data_loader)
        tb_write.add_scalar("test_loss", eval_loss, global_step)
        tb_write.add_scalar("test_acc", eval_acc, global_step)
        print("test_loss: {}, test_acc:{}".format(eval_loss, eval_acc))
        model.train()

        # 每个epoch进行完，则保存模型
        output_dir = os.path.join(args.output_dir, "epoch_{}.bin".format(epoch))
        torch.save(model.state_dict(), output_dir)

