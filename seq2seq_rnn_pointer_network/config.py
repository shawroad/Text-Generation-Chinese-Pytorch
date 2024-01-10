"""
@file   : config.py
@author : Henry
@email  : luxiaonlp@163.com
@time   : 2024-01-02
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output_dir', default='output_dir/', type=str, help='模型输出路径')
    parser.add_argument('--train_data_src', default='../data/train.src.txt', type=str, help='训练文章')
    parser.add_argument('--train_data_tgt', default='../data/train.tgt.txt', type=str, help='训练摘要')
    parser.add_argument('--valid_data_src', default='../data/test.src.txt', type=str, help='验证文章')
    parser.add_argument('--valid_data_tgt', default='../data/test.tgt.txt', type=str, help='验证摘要')

    parser.add_argument('--test_data_src', default='../data/test.src.txt', type=str, help='测试文章')
    parser.add_argument('--test_data_tgt', default='../data/test.tgt.txt', type=str, help='测试摘要')
    parser.add_argument('--cov_lambda', default=1, type=int)

    parser.add_argument('--vocab2id_path', default='../data/vocab2id.json', type=str, help='词表')
    parser.add_argument('--emb_size', default=256, type=int, help='词嵌入大小')
    parser.add_argument('--hidden_size', default=512, type=int, help='隐层大小')
    parser.add_argument('--num_layers', default=2, type=int, help='层大小')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout')
    parser.add_argument('--num_train_epochs', default=50, type=int, help='模型训练的轮数')
    parser.add_argument('--batch_size', default=64, type=int, help='训练时每个batch的大小')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='模型训练时的学习率')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度积累')
    parser.add_argument('--pointer', default=False, type=bool, help='是否只用指针网络')
    parser.add_argument('--logging_steps', default=5, type=int, help='保存训练日志的步数')
    return parser.parse_args()
