"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-08-04
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser('--小作文生成')
    parser.add_argument('--batch_size', default=8, type=int, help='批次大小')
    parser.add_argument('--pretrain_model_path', default='./t5_pretrain', type=str, help='预训练模型')
    parser.add_argument('--data_path', default='../data/article.json', type=str, help='处理过的训练数据')
    parser.add_argument('--epochs', default=10, type=int, required=False, help='训练的轮次')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--learning_rate', default=2e-4, type=float, required=False, help='学习率')
    parser.add_argument('--output_dir', default='./output', type=str, required=False, help='多少步汇报一次loss')
    parser.add_argument('--warmup_steps_rate', default=0.005, type=float, required=False, help='warm up步数占总步数的比例')
    return parser.parse_args()
