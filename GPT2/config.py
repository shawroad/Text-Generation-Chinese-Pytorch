"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-08-03
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser('--小作文生成')
    parser.add_argument('--pretrain_model_path', default='./gpt2_pretrain', type=str, help='预训练模型')
    parser.add_argument('--pretrain_model_config_path', default='./gpt2_pretrain/config.json', type=str, help='预训练模型')
    parser.add_argument('--train_data_path', default='../data/article.json', type=str, help='训练数据')
    parser.add_argument('--train_data_path_processed', default='./data/train.json', type=str, help='训练数据')
    parser.add_argument('--output_dir', default='./output', type=str, required=False, help='多少步汇报一次loss')

    parser.add_argument('--max_len', default=200, type=int, required=False, help='输入的最大长度')
    parser.add_argument('--batch_size', default=16, type=int, required=False, help='训练batch size')
    parser.add_argument('--epochs', default=50, type=int, required=False, help='训练的轮次')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--learning_rate', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps_rate', default=0.05, type=float, required=False, help='warm up步数占总步数的比例')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    return parser.parse_args()
