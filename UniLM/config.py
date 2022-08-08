"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-08-05
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser('--小作文生成')
    parser.add_argument('--corpus_path', default='../data/article.json', type=str, help='训练数据')
    parser.add_argument('--bert_vocab_path', default='./roberta_pretrain/vocab.txt', type=str, help='bert的词表')
    parser.add_argument('--bert_pretrain_weight_path', default='./roberta_pretrain/pytorch_model.bin', type=str, help='unilm的权重')

    parser.add_argument('--output_dir', default='./outputs', type=str, help='模型的输出')
    parser.add_argument('--batch_size', default=16, type=int, help='训练批次的大小')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='学习率的大小')
    parser.add_argument('--warmup_proportion', default=0.01, type=float, help='warm up概率，即训练总步长的百分之多少，进行warm up')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度积累')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Adam优化器的epsilon值')
    parser.add_argument('--max_length', default=300, type=int, help='最大长度')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='训练几轮')
    return parser.parse_args()
