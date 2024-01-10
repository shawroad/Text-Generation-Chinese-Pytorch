"""
@file   : model.py
@author : Henry
@email  : luxiaonlp@163.com
@time   : 2024-01-06
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Embeddings(nn.Module):
    """
    Implements embeddings of the words and adds their positional encodings.
    """
    def __init__(self, vocab_size, d_model, max_len=256):
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(0.1)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = self.create_positinal_encoding(max_len, self.d_model)
        self.dropout = nn.Dropout(0.1)

    def create_positinal_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model).to(device)
        for pos in range(max_len):  # for each position of the word
            for i in range(0, d_model, 2):  # for each dimension of the each position
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)  # include the batch size
        return pe

    def forward(self, encoded_words):
        embedding = self.embed(encoded_words) * math.sqrt(self.d_model)
        embedding += self.pe[:, :embedding.size(1)]  # pe will automatically be expanded with the same batch size as encoded_words
        embedding = self.dropout(embedding)
        return embedding


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super(MultiHeadAttention, self).__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = nn.Dropout(0.1)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.concat = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask):
        """
        query, key, value of shape: (batch_size, max_len, 512)
        mask of shape: (batch_size, 1, 1, max_words)
        """
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        # print(query.size(), key.size(), value.size())
        # torch.Size([2, 8, 512]) torch.Size([2, 8, 512]) torch.Size([2, 8, 512])
        # query: (batch_size, max_len, 512) -> (batch_size, max_len, 8, 64) -> (batch_size, 8, max_len,  64)
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        # (batch_size, 8, max_len, 64)
        # (batch_size, 8, max_len, max_len)

        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query.size(-1))
        # (batch_size, 8, max_len, max_len)

        # print(mask.size())   # torch.Size([2, 1, 1, 14])
        # [1, 1, 1, 0, 0, 0]
        scores = scores.masked_fill(mask == 0, -1e9)  # (batch_size, h, max_len, max_len)
        weights = F.softmax(scores, dim=-1)  # (batch_size, h, max_len, max_len)

        weights = self.dropout(weights)
        # print(weights.size())   # torch.Size([8, 8, 118, 118])

        # (batch_size, h, max_len, max_len) matmul (batch_size, h, max_len, d_k) --> (batch_size, h, max_len, d_k)
        context = torch.matmul(weights, value)
        # (batch_size, h, max_len, d_k) --> (batch_size, max_len, h, d_k) --> (batch_size, max_len, h * d_k)
        context = context.permute(0, 2, 1, 3).contiguous().view(context.shape[0], -1, self.heads * self.d_k)
        # batch_size, max_len, hidden_size
        # print(context.size())    # torch.Size([8, 9, 512])
        interacted = self.concat(context)
        return interacted, weights


class FeedForward(nn.Module):
    def __init__(self, d_model, middle_dim=2048):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, middle_dim)
        self.fc2 = nn.Linear(middle_dim, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()
        self.layernorm = nn.LayerNorm(d_model)   # d_model
        self.self_multihead = MultiHeadAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, embeddings, mask):
        # embeddings: batch_size, max_len, 512
        x, attn = self.self_multihead(embeddings, embeddings, embeddings, mask)
        interacted = self.dropout(x)
        # batch_size, max_len, hidden_size

        interacted = self.layernorm(interacted + embeddings)   # 残差+归一化
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        # feed_forward_out: batch_size, max_len, hidden_size
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(DecoderLayer, self).__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadAttention(heads, d_model)
        self.src_multihead = MultiHeadAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, embeddings, encoded, src_mask, target_mask):
        x, _ = self.self_multihead(embeddings, embeddings, embeddings, target_mask)
        query = self.dropout(x)
        query = self.layernorm(query + embeddings)

        x, attn = self.src_multihead(query, encoded, encoded, src_mask)
        interacted = self.dropout(x)
        interacted = self.layernorm(interacted + query)
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        decoded = self.layernorm(feed_forward_out + interacted)
        return decoded, attn


class Transformer(nn.Module):
    def __init__(self, d_model, heads, num_layers, word_map):
        '''
        d_model: 512
        heads: 8
        num_layers: 6
        word_map: vocab2id
        '''
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.vocab_size = len(word_map)
        self.embed = Embeddings(self.vocab_size, d_model)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, heads) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, heads) for _ in range(num_layers)])
        self.logit = nn.Linear(d_model, self.vocab_size)

        # Pointer work 指针网络
        self.switch = nn.Linear(self.vocab_size, 1)

    def encode(self, src_words, src_mask):
        src_embeddings = self.embed(src_words)
        # print(src_embeddings.size())   # (batch_size,max_len,512)

        for layer in self.encoder:
            src_embeddings = layer(src_embeddings, src_mask)
        return src_embeddings

    def decode(self, target_words, target_mask, src_embeddings, src_mask):
        tgt_embeddings = self.embed(target_words)
        # print(tgt_embeddings.size())

        for layer in self.decoder:
            tgt_embeddings, attention = layer(tgt_embeddings, src_embeddings, src_mask, target_mask)
    
        return tgt_embeddings


    def forward(self, src_words, src_mask, target_words, target_mask):
        encoded = self.encode(src_words, src_mask)
        # print(encoded.size())   # batch_size, max_len, hidden_size

        # print(encoded.size())    # torch.Size([2, 6, 512])
        # [SOS, xx, xxx, ] -》[2, xx,xx]
        
        out = self.decode(target_words, target_mask, encoded, src_mask)
        out = F.log_softmax(self.logit(out), dim=2)
        return out
        

