"""
@file   : run_train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-08-04
"""
import os
import torch.cuda
from datetime import datetime
from torch.optim import AdamW
from config import set_args
from torch.nn import CrossEntropyLoss
from tokenizer import T5PegasusTokenizer
from torch.utils.data import DataLoader
from data_helper import MT5Dataset, load_data, collate_fn
from transformers import get_linear_schedule_with_warmup
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration


def calc_loss(logits, decoder_input_ids, decoder_attention_mask):
    # 计算损失
    decoder_mask = decoder_attention_mask[:, 1:].reshape(-1).bool()
    logits = logits[:, :-1]
    logits = logits.reshape((-1, logits.size(-1)))[decoder_mask]
    labels = decoder_input_ids[:, 1:].reshape(-1)[decoder_mask]
    loss = loss_fct(logits, labels)
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


if __name__ == '__main__':
    args = set_args()
    tokenizer = T5PegasusTokenizer.from_pretrained(args.pretrain_model_path)
    train_df = load_data(args.data_path)
    train_dataset = MT5Dataset(train_df, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    model = MT5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)

    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('total parameters nums:', num_parameters)

    if torch.cuda.is_available():
        model.cuda()

    total_steps = int(len(train_dataset) * args.epochs / args.batch_size / args.gradient_accumulation)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps_rate * total_steps,
                                                num_training_steps=total_steps)

    loss_fct = CrossEntropyLoss(ignore_index=0)

    for epoch in range(args.epochs):
        model.train()
        epoch_start_time = datetime.now()

        epoch_loss, epoch_acc = 0, 0
        for step, batch in enumerate(train_dataloader):
            if torch.cuda.is_available():
                batch = (t.cuda() for t in batch)
            input_ids, input_mask, decoder_input_ids, decoder_attention_mask = batch
            output = model(input_ids=input_ids, attention_mask=input_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask)
            logits = output.logits

            loss = calc_loss(logits, decoder_input_ids, decoder_attention_mask)
            accuracy = calculate_acc(logits, decoder_input_ids, ignore_index=tokenizer.pad_token_id)

            loss.backward()
            print('epoch:{}, step:{}, loss:{:10f}, accuracy:{:10f}, lr:{:10f}'.format(epoch, step, loss, accuracy, scheduler.get_last_lr()[0]))
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            epoch_loss += loss.item()
            epoch_acc += accuracy

            if (step + 1) % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  # 进行warm_up

        avg_loss = epoch_loss / len(train_dataloader)
        avg_acc = epoch_acc / len(train_dataloader)

        ss = 'epoch:{}, loss:{:10f}, accuracy:{:10f}'.format(epoch, avg_loss, avg_acc)
        loss_path = os.path.join(args.output_dir, 'loss.txt'.format(epoch + 1))
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
