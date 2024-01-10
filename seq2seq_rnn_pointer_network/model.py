"""
@file   : model.py
@author : Henry
@email  : luxiaonlp@163.com
@time   : 2024-01-02
"""
import torch
import random
from torch import nn
from config import set_args
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


args = set_args()


class Encoder(nn.Module):
    def __init__(self, vocab):
        super(Encoder, self).__init__()
        self.vocab_size = len(vocab)
        self.gru = nn.GRU(args.emb_size, args.hidden_size, num_layers=args.num_layers, batch_first=True, dropout=args.dropout)
        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.hidden_size, args.hidden_size)
        self.relu = nn.ReLU()

    def forward(self, enc_input, text_lengths):
        text_lengths = text_lengths.to("cpu")
        # print(text_lengths)
        # embedded = self.dropout(self.embedding(enc_input))  # [batch_size, seq_len, emb_size]
        embedded = self.dropout(enc_input)
        embedded = pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.gru(embedded)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.relu(self.linear(output))
        return output, hidden[-1].detach()


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.linear = nn.Linear(args.hidden_size * 2 + args.emb_size, args.hidden_size)
        self.v = nn.Linear(args.hidden_size, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, dec_input, enc_output, text_lengths, coverage_vector):
        # print(dec_input.size(), enc_output.size(), text_lengths.size(), coverage_vector.size())
        # torch.Size([8, 1, 128]) torch.Size([8, 107, 256]) torch.Size([8]) torch.Size([8, 107])
        
        seq_len, hidden_size = enc_output.size(1), enc_output.size(-1)

        s = dec_input.repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_size] torch.Size([8, 107, 128])
        # print(s.size())   # torch.Size([8, 107, 128])
        coverage_vector_copy = coverage_vector.unsqueeze(2).repeat(1, 1, hidden_size)

        x = torch.tanh(self.linear(torch.cat([enc_output, s, coverage_vector_copy], dim=2)))
        # print(x.size())   # torch.Size([8, 107, 256])

        attention = self.v(x).squeeze(-1)  # [batch_size, seq_len]
        max_len = enc_output.size(1)

        text_lengths = text_lengths.to('cpu')
        mask = torch.arange(max_len).expand(text_lengths.shape[0], max_len) >= text_lengths.unsqueeze(1)
        
        attention.masked_fill_(mask.to(dec_input.device), float('-inf'))
        attention_weights = self.softmax(attention)
        # 更新 coverage_vector
        coverage_vector += attention_weights
        return attention_weights, coverage_vector  # [batch, seq_len], [batch_size, seq_len]


class Decoder(nn.Module):
    def __init__(self, vocab, attention):
        super(Decoder, self).__init__()
        self.vocab_size = len(vocab)
        # self.embedding = nn.Embedding(self.vocab_size, args.emb_size, padding_idx=vocab['<PAD>'])
        self.attention = attention
        self.gru = nn.GRU(args.emb_size + args.hidden_size, args.hidden_size, batch_first=True)
        self.linear = nn.Linear(args.emb_size + 2 * args.hidden_size, self.vocab_size)
        self.dropout = nn.Dropout(args.dropout)

        # PGN 网络  计算p_gen
        self.w_gen = nn.Linear(args.hidden_size * 2 + args.emb_size, 1)

    def forward(self, dec_input, prev_hidden, enc_output, text_lengths, coverage_vector):
        # print(prev_hidden.size())   # torch.Size([8, 256])
        # print(enc_output.size())   # torch.Size([8, 107, 256])
        # embedded = self.embedding(dec_input)
        embedded = dec_input

        attention_weights, coverage_vector = self.attention(embedded, enc_output, text_lengths, coverage_vector)
        # print(attention_weights.size())   # torch.Size([8, 107])
        # print(coverage_vector.size())   # torch.Size([8, 107])

        attention_weights = attention_weights.unsqueeze(1)   # [batch_size, 1, enc_len]
        c = torch.bmm(attention_weights, enc_output)   # [batch_size, 1, hidden_size]

        # 将经过 embedding 处理过的 decoder 输入，和上下文向量一起送入到 GRU 网络中
        gru_input = torch.cat([embedded, c], dim=2)
        # print(gru_input.size())   # torch.Size([8, 1, 384])

        # prev_hidden 是上个时间步的隐状态，作为 decoder 的参数传入进来
        dec_output, dec_hidden = self.gru(gru_input, prev_hidden.unsqueeze(0))
        dec_output = self.linear(torch.cat((dec_output.squeeze(1), c.squeeze(1), embedded.squeeze(1)), dim=1)) # [batch_size, vocab_size]
        dec_hidden = dec_hidden.squeeze(0)
        x_gen = torch.cat([dec_hidden, c.squeeze(1), embedded.squeeze(1)], dim=1)
        p_gen = torch.sigmoid(self.w_gen(x_gen))
        return dec_output, dec_hidden, attention_weights.squeeze(1), p_gen, coverage_vector


class Model(nn.Module):
    # PGN网络实现
    def __init__(self, vocab):
        super(Model, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding = nn.Embedding(self.vocab_size, args.emb_size, padding_idx=vocab["<PAD>"])
        
        self.encoder = Encoder(vocab)

        attention = Attention()
        self.decoder = Decoder(vocab, attention)

    def get_final_distribution(self, x, p_gen, p_vocab, attention_weights, max_oov):
        batch_size = x.shape[0]
        p_gen = torch.clamp(p_gen, 0.001, 0.999)
        p_vocab_weighted = p_gen * p_vocab
        attention_weighted = (1 - p_gen) * attention_weights
        # 加入 max_oov 维度，将原文中的 OOV 单词考虑进来
        extension = torch.zeros((batch_size, max_oov), dtype=torch.float).to(x.device)
        p_vocab_extended = torch.cat([p_vocab_weighted, extension], dim=-1)
        # p_gen * p_vocab + (1 - p_gen) * attention_weights, 将 attention weights 中的每个位置 idx 映射成该位置的 token_id
        final_distribution = p_vocab_extended.scatter_add_(dim=1, index=x, src=attention_weighted)
        # 输出最终的 vocab distribution [batch_size, vocab_size + len(oov)]
        return final_distribution

    def forward(self, src, tgt, src_lengths, teacher_forcing_ratio=0.5):
        src_emb = self.embedding(src)
        # batch_size, max_len
        enc_output, prev_hidden = self.encoder(src_emb, src_lengths)
    
        # print(enc_output.size())   # torch.Size([8, 115, 256])
        # print(prev_hidden.size())   # torch.Size([8, 256])
        batch_size = tgt.size(0)
        tgt_len = tgt.size(1)

        dec_input = tgt[:, 0]
        dec_outputs = torch.zeros(batch_size, tgt_len, len(self.vocab)).to(src.device)
        coverage_vector = torch.zeros_like(src, dtype=torch.float32).to(src.device)

        for t in range(tgt_len - 1):
            dec_input = dec_input.unsqueeze(1)  # torch.Size([8, 1])
            dec_emb = self.embedding(dec_input)

            # dec_output, prev_hidden, _ = self.decoder(dec_emb, prev_hidden, enc_output, src_lengths)
            dec_output, prev_hidden, attention_weights, p_gen, coverage_vector = self.decoder(dec_emb, prev_hidden,
                                                                                              enc_output, src_lengths,
                                                                                              coverage_vector)
            final_distribution = self.get_final_distribution(src, p_gen, dec_output, attention_weights, 0)
            teacher_force = random.random() < teacher_forcing_ratio
            dec_outputs[:, t, :] = final_distribution
            top1 = final_distribution.argmax(1)
            dec_input = tgt[:, t] if teacher_force else top1
        return dec_outputs, attention_weights, coverage_vector

