"""
@file   : inference.py
@author : Henry
@email  : luxiaonlp@163.com
@time   : 2024-01-06
"""
import json
import torch
import torch.utils.data
from model import Transformer
from rouge import Rouge
from config import set_args
from tqdm.contrib import tzip
from data_helper import load_data
import torch.nn.functional as F
import pandas as pd




@torch.no_grad()
def predict(model, vocab2id, context):
    input_text = str(context)
    input_ids = [vocab2id.get(v, vocab2id['<UNK>']) for v in list(input_text)]
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    input_mask = (input_ids != 0).unsqueeze(1).unsqueeze(1)

    start_token = vocab2id['<SOS>']
    if torch.cuda.is_available():
        input_ids, input_mask = input_ids.cuda(), input_mask.cuda()

    encoded = model.encode(input_ids, input_mask)
        
    words = torch.tensor([[vocab2id['<SOS>']]])
    for step in range(max_len):
        size = words.size(1)
        # 下三角矩阵
        target_mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        target_mask = target_mask.unsqueeze(0).unsqueeze(0)

        # 下三角
        if torch.cuda.is_available():
            words, target_mask = words.cuda(), target_mask.cuda()
        
        decoded = model.decode(words, target_mask, encoded, input_mask)
        predictions = model.logit(decoded[:, -1])
        _, next_word = torch.max(predictions, dim=1)
        
        next_word = next_word.item()
        if next_word == vocab2id['<EOS>']:
            break
        words = torch.cat([words, torch.LongTensor([[next_word]]).cuda()], dim=1)  # (1,step+2)

    # [1, 128] => [128]   .squeeze(0)
    if words.dim() == 2:
        words = words.squeeze(0)
        words = words.tolist()

    sen_idx = [w for w in words if w not in {vocab2id['<SOS>']}]
    sentence = ''.join([id2vocab[sen_idx[k]] for k in range(len(sen_idx))])
    return sentence



if __name__ == '__main__':
    args = set_args()
    test_context = load_data(args.test_data_src)[:100]
    test_summary = load_data(args.test_data_tgt)[:100]
    vocab2id = json.load(open(args.vocab2id_path, 'r', encoding='utf8'))
    max_len = 60

    id2vocab = {id: vocab for vocab, id in vocab2id.items()}
    model = Transformer(d_model=args.d_model, heads=args.heads, num_layers=args.num_layers, word_map=vocab2id)
    model.load_state_dict(torch.load('./output/epoch_{}.bin'.format(0), map_location='cpu'))
    if torch.cuda.is_available():
        model.cuda()
    
    model.eval()
    
    final_context, final_summary, final_gen_summary = [], [], []
    for context, summary in tzip(test_context, test_summary):
        gen_summary = predict(model, vocab2id, context)
        final_context.append(context)
        final_summary.append(summary)
        final_gen_summary.append(gen_summary)
    df = pd.DataFrame({'context': final_context, 'summary': final_summary, 'gen_summary': final_gen_summary})
    df.to_csv('./result.csv', index=False)
    
    # 计算指标
    rouge = Rouge()
    hyps, refs = [], []
    for context, summary, gen_summary in zip(df['context'], df['summary'], df['gen_summary']):
        refs.append(' '.join(list(summary)))
        hyps.append(' '.join(list(gen_summary)))
    scores = rouge.get_scores(hyps, refs, avg=True)
    print(scores)



