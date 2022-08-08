"""
@file   : inference.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-02-22
"""
import torch
from model import Model
from config import set_args
import torch.nn.functional as F
from bert_model import BertConfig
from tokenizer import load_bert_vocab, Tokenizer


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1   # 注意 这里只能支持单样本的处理
    top_k = min(top_k, logits.size(-1))   # 也算是一个检查吧，logits.size(-1)说明你有多少个vocab, 如果你top_k都大于vocab_num, 那topk还有意义吗，所以选最小的 
   
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]  # 首先得到topk个最大概率中的最小概率。 然后小于这个最小概率的，都是我们不在我们采样的序列中
        logits[indices_to_remove] = filter_value   # 将这些不在采样序列中的token的概率置成负无穷
    
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits递减排序 返回两个东西: 1. 排序后的概率序列 2. 排序后每个概率原先在的位置
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)  # 对排序后的概率softmax,然后累加 
        # 举个例子说明上面torch.cumsum()函数   假设有序列 [5, 4, 2, 1] 执行torch.cumsum()后变成: [5, 9, 11, 12]

        sorted_indices_to_remove = cumulative_probs > top_p  # 输出: [False, Flase, True, True, True, True]
       
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()  # 将索引向右移动，使第一个位置保持在阈值之上
        sorted_indices_to_remove[..., 0] = 0   # 至少要确保概率最高的满足情况  不做这一步处理可能通过卡阈值 当前的概率都不满足
      
        indices_to_remove = sorted_indices[sorted_indices_to_remove]   # 取出不在采样序列中的token索引
        logits[indices_to_remove] = filter_value   # 然后将其概率置为负无穷
    return logits


def sample_decode(encode_output):
    # {'token_ids': [101, 2769, 4638, 1959, 1959, 102, 2769, 4638, 1959, 1959, 3221, 671, 702, 1249, 1227, 3320, 2141, 4638, 1093, 3333, 1967, 1957, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
        
    max_len = 50  # 最多生成50个token
    repetition_penalty = 1.1  # 惩罚项 主要是为了让生成的token在后面尽量少出现
    temperature = 1   # 控制样本的生成尽可能多样性
    topk = 5  # 最高k选1
    topp = 0.95   # 最高累计概率
    
    input_ids, token_type_ids = encode_output['token_ids'], encode_output['token_type_ids']
    curr_input_tensor = torch.tensor([input_ids]).long()   # 把输入整理成: [CLS] input [SEP]
    token_type_ids = torch.tensor([token_type_ids]).long()
    if torch.cuda.is_available():
        curr_input_tensor = curr_input_tensor.cuda()
        token_type_ids = token_type_ids.cuda()
    
    generated = []
    for _ in range(max_len):
        with torch.no_grad():
            outputs = model(curr_input_tensor, token_type_ids)
        print(outputs.size())   # torch.Size([1, 4, 50000]) 

        next_token_logits = outputs[0][-1, :]  # 这里相当于取得是序列最后的位置的logits  我们认为其是预测的下一个token的概率
        # print(next_token_logits.size())

        # 如果某个token之前已经生成过了，给其一个惩罚 就是在其对应的概率上出一个惩罚因子, 显然这个惩罚因子要>=1.
        for id in set(generated):
            next_token_logits[id] /= repetition_penalty

        # 这里对所有的概率都处于一个temperature,是为了将概率整体的放小，然后通过softmax后，它们之间的差距就不是很大，
        # 这样低概率的就更有可能采样到。满足其多样性
        next_token_logits = next_token_logits / temperature
        # print(next_token_logits.size())

        # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
        next_token_logits[word2idx['[UNK]']] = -float('Inf') 

        # 可以指定topk  也可以指定topp,  topk是选概率最大的前k个做采样， topp卡的时累计概率，如果越大，说明你采样的序列越多，概率越小，采样序列越小。
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=topk, top_p=topp)

        # 按概率采样  概率越高 越容易被采样到. 
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)    # 按概率采样 采出当前解码出的token索引

        if next_token == sep_token_id:  # 遇到[SEP]则表明response生成结束
            break

        generated.append(next_token.item())
        
        next_token = next_token.unsqueeze(-1)
        next_token_type_ids = torch.tensor([[1]], dtype=torch.long)
        if torch.cuda.is_available():
            next_token_type_ids = next_token_type_ids.cuda()
        curr_input_tensor = torch.cat((curr_input_tensor, next_token), dim=1)
        token_type_ids = torch.cat((token_type_ids, next_token_type_ids), dim=1)
        
    result = []
    for t in generated:
        result.append(idx2word.get(t, word2idx['[UNK]']))
    result = ''.join(result)
    result = result.replace('[UNK]', '')
    return result
    

if __name__ == '__main__':
    args = set_args()
    word2idx = load_bert_vocab(args.bert_vocab_path)
    idx2word = {}
    for word, idx in word2idx.items():
        idx2word[idx] = word

    cls_token_id = word2idx['[CLS]']
    sep_token_id = word2idx['[SEP]']
    tokenizer = Tokenizer(word2idx)

    config = BertConfig(len(word2idx))
    model = Model(config)
    
    # 加载模型
    model.load_state_dict(torch.load('./checkpoint/Epoch-9.bin', map_location='cpu'))
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    title = '我的奶奶'
    prefix = '我的奶奶是一个勤劳朴实的农村妇女'
    token_ids, token_type_ids = tokenizer.encode(title, prefix)
    output = {
        "token_ids": token_ids,
        "token_type_ids": token_type_ids,
    }
    # {'token_ids': [101, 2769, 4638, 1959, 1959, 102, 2769, 4638, 1959, 1959, 3221, 671, 702, 1249, 1227, 3320, 2141, 4638, 1093, 3333, 1967, 1957, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    
    res = sample_decode(output)
    print(res)

