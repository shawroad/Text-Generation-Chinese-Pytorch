"""
@file   : config.py
@author : Henry
@email  : luxiaonlp@163.com
@time   : 2024-01-06
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser('--rnn gen')
    parser.add_argument('--epochs', default=50, type=int, help='训练几轮')
    parser.add_argument('--learning_rate', default=1e-3, type=int, help='学习率')
    parser.add_argument('--batch_size', default=64, type=int, help='训练的批次大小')

    parser.add_argument('--train_data_src', default='../data/train.src.txt', type=str, help='训练文章')
    parser.add_argument('--train_data_tgt', default='../data/train.tgt.txt', type=str, help='训练摘要')
    parser.add_argument('--valid_data_src', default='../data/test.src.txt', type=str, help='验证文章')
    parser.add_argument('--valid_data_tgt', default='../data/test.tgt.txt', type=str, help='验证摘要')
    parser.add_argument('--test_data_src', default='../data/test.src.txt', type=str, help='测试文章')
    parser.add_argument('--test_data_tgt', default='../data/test.tgt.txt', type=str, help='测试摘要')
    parser.add_argument('--vocab2id_path', default='../data/vocab2id.json', type=str, help='词表')
    parser.add_argument('--logging_steps', default=5, type=int)

    # clip
    parser.add_argument('--clip', default=2.0, type=float, help='梯度裁剪')
    parser.add_argument('--seed', default=43, type=int, help='随机种子大小')
    parser.add_argument('--output_dir', default='./output', type=str, help='模型输出路径')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度积聚')

    parser.add_argument('--hidden_size', default=1024, type=int, help='隐层大小')
    parser.add_argument('--d_model', default=512, type=int, help='')
    parser.add_argument('--heads', default=8, type=int, help='')
    parser.add_argument('--num_layers', default=6, type=int, help='')
    return parser.parse_args()
