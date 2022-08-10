"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-08-03
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help='设置随机种子')
    parser.add_argument('--train_data_path', default='../data/article.json', type=str, required=False, help='经过预处理之后的数据存放路径')
    parser.add_argument('--train_data_path_processed', default='./data/train.json', type=str, required=False, help='经过预处理之后的数据存放路径')
    parser.add_argument('--cpm_model_config', default='./cpm_pretrain/cpm-small.json', type=str, help='模型配置')
    parser.add_argument('--cpm_model_vocab', default='./cpm_pretrain/chinese_vocab.model', type=str, help='cpm模型的词表')

    parser.add_argument('--max_len', default=200, type=int, required=False, help='训练数据最大长度')
    parser.add_argument('--train_batch_size', default=16, type=int, required=False, help='训练的batch size')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, required=False, help='梯度积累的步数')
    parser.add_argument('--epochs', default=50, type=int, required=False, help='训练的最大轮次')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--eps', default=1.0e-09, type=float, required=False, help='AdamW优化器的衰减率')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='warm up步数')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--output_dir', default='./output', type=str, required=False, help='输出路径')
    args = parser.parse_args()
    return args
