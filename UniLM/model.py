"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-08-05
"""
import torch
import torch.nn as nn
from bert_model import BertModel, BertConfig, BertLMPredictionHead
from tokenizer import Tokenizer, load_bert_vocab
from config import set_args


args = set_args()


class Model(nn.Module):
    def __init__(self, config: BertConfig):
        super(Model, self).__init__()
        # 获取配置信息
        self.hidden_dim = config.hidden_size
        self.vocab_size = config.vocab_size

        self.bert = BertModel(config)

        # 解码
        self.decoder = BertLMPredictionHead(config, self.bert.embeddings.word_embeddings.weight)

        # 加载字典和分词器
        self.word2ix = load_bert_vocab(args.bert_vocab_path)
        self.tokenizer = Tokenizer(self.word2ix)

    def compute_loss(self, predictions, labels, target_mask):
        """
        target_mask : 句子a部分和pad部分全为0， 而句子b部分为1
        """
        predictions = predictions.view(-1, self.vocab_size)
        labels = labels.view(-1)
        target_mask = target_mask.view(-1).float()
        loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        return (loss(predictions, labels) * target_mask).sum() / target_mask.sum()  # 通过mask 取消 pad 和句子a部分预测的影响

    def forward(self, input_tensor, token_type_id=None, position_enc=None, labels=None):
        input_shape = input_tensor.size()   # batch_size, max_len
        seq_len = input_shape[1]
        # 构建特殊的mask
        if torch.cuda.is_available():
            ones = torch.ones((1, 1, seq_len, seq_len), dtype=torch.float32).cuda()
        else:
            ones = torch.ones((1, 1, seq_len, seq_len), dtype=torch.float32)

        a_mask = ones.tril()  # 下三角矩阵

        s_ex12 = token_type_id.unsqueeze(1).unsqueeze(2).float()  # batch_size, 1, 1, max_len
        s_ex13 = token_type_id.unsqueeze(1).unsqueeze(3).float()  # batch_size, 1, max_len, 1

        a_mask = (1.0 - s_ex12) * (1.0 - s_ex13) + s_ex13 * a_mask

        # print(a_mask.size())    # torch.Size([2, 1, 33, 33])
        enc_layers, _ = self.bert(input_tensor,
                                  position_ids=position_enc,
                                  token_type_ids=token_type_id,
                                  attention_mask=a_mask,
                                  output_all_encoded_layers=True)

        sequence_out = enc_layers[-1]  # 取出来最后一层输出
        # print(sequence_out.size())   # torch.Size([2, 267, 768])

        predictions = self.decoder(sequence_out)

        if labels is not None:
            # 计算loss 需要构建特殊的输出mask 才能计算正确的loss
            predictions = predictions[:, :-1].contiguous()    # 错位预测  那最后以为的预测 就没有意义 也就不用进行损失计算
            target_mask = token_type_id[:, 1:].contiguous()   # 从第二位开始  计算loss
            loss = self.compute_loss(predictions, labels, target_mask)

            return predictions, loss
        else:
            return predictions

    def generate(self, text, out_max_length=10, beam_size=3):
        token_ids, token_type_ids = self.tokenizer.encode(text)
        token_ids = torch.tensor(token_ids).view(1, -1)
        token_type_ids = torch.tensor(token_type_ids).view(1, -1)
        if torch.cuda.is_available():
            token_ids = token_ids.cuda()
            token_type_ids = token_type_ids.cuda()

        output_ids = self.beam_search(token_ids, token_type_ids, self.word2ix, beam_size=beam_size,
                                      device=token_ids.device, out_max_length=out_max_length)
        return self.tokenizer.decode(output_ids)

    def beam_search(self, token_ids, token_type_ids, word2ix, beam_size=1, device='cpu', out_max_length=10):
        sep_id = word2ix['[SEP]']
        # 用来保存输出序列
        output_ids = torch.empty(1, 0, device=device, dtype=torch.long)
        with torch.no_grad():
            output_scores = torch.zeros(token_ids.shape[0], device=device)
            for step in range(out_max_length):
                if step == 0:
                    scores = self.forward(token_ids, token_type_ids)
                    # 重复beam_size次
                    token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                    token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
                else:
                    scores = self.forward(new_input_ids, new_token_type_ids)

                logit_score = torch.log_softmax(scores[:, -1], dim=-1)  # 取出当前步对所有词的预测
                logit_score = output_scores.view(-1, 1) + logit_score   # 累计概率

                # 展开  取topk
                logit_score = logit_score.view(-1)
                hype_score, hype_pos = torch.topk(logit_score, beam_size)
                # print(hype_score)   # tensor([-5.6436, -5.7821, -5.8964])
                # print(hype_pos)   # tensor([4743, 4131, 2115])
                indice1 = torch.div(hype_pos,  scores.shape[-1], rounding_mode='floor')
                indice2 = (hype_pos % scores.shape[-1]).long().reshape(-1, 1)   # 列索引

                # 更新分数
                output_scores = hype_score
                output_ids = torch.cat([output_ids[indice1], indice2], dim=1).long()
                new_input_ids = torch.cat([token_ids, output_ids], dim=1)   # 下一步的输入
                new_token_type_ids = torch.cat([token_type_ids, torch.ones_like(output_ids)], dim=1)
                end_counts = (output_ids == sep_id).sum(1)   # 统计出现end的标记
                best_one = output_scores.argmax()
                if end_counts[best_one] == 1:
                    # 当前概率累加最大的  出现了结束标记  那就终止了
                    return output_ids[best_one][:-1]
                else:
                    # 保留未完成的部分
                    flag = (end_counts < 1)   # 标记未完成的序列
                    if not flag.all():
                        # 如果有已经完成的
                        token_ids = token_ids[flag]
                        token_type_ids = token_type_ids[flag]

                        new_input_ids = new_input_ids[flag]
                        new_token_type_ids = new_token_type_ids[flag]

                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        end_counts = end_counts[flag]  # 扔掉已完成end计数
                        beam_size = flag.sum()  # topk相应变化
            return output_ids[output_scores.argmax()]
