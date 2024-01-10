"""
@file   : inference.py
@author : Henry
@email  : luxiaonlp@163.com
@time   : 2024-01-02
"""
import os
import json
import torch
import random
import pandas as pd
from model import Model
from rouge import Rouge
from config import set_args
from tqdm.contrib import tzip
from data_helper import load_data
from transformers.models.bert.tokenization_bert import BasicTokenizer


# 模型预测过程
def predict(model, vocab, text, max_len=60):
    model.eval()
    with torch.no_grad():
        # context_token = tokenizer.tokenize(text)
        context_token = list(text)
        context_input_ids = []
        for word in context_token:
            context_input_ids.append(vocab2id.get(word, vocab2id.get('<UNK>')))

        src_lengths = torch.tensor([len(context_input_ids)])
        src_input_ids = torch.tensor([context_input_ids])
        if torch.cuda.is_available():
            src_lengths = src_lengths.cuda()
            src_input_ids = src_input_ids.cuda()


        src_emb = model.embedding(src_input_ids)

        coverage_vector = torch.zeros_like(src_input_ids, dtype=torch.float32).to(src_emb.device)
            
        enc_output, prev_hidden = model.encoder(src_emb, src_lengths)
    
        # enc_output, prev_hidden = model.encoder(src_input_ids, src_lengths)
        # print(enc_output.size())   # torch.Size([1, 94, 256])
        decoder_input_ids = torch.tensor([vocab['<SOS>']]).to(src_input_ids.device)
        result = []
        for t in range(max_len):
            dec_input = decoder_input_ids.unsqueeze(1) 
            decoder_emb = model.embedding(dec_input)
                

            dec_output, prev_hidden, attention_weights, p_gen, coverage_vector = model.decoder(decoder_emb, prev_hidden,
                                                                                              enc_output, src_lengths,
                                                                                              coverage_vector)
    
            final_distribution = model.get_final_distribution(src_input_ids, p_gen, dec_output, attention_weights, 0)
            max_ids = final_distribution.argmax(1)
                
            # _, max_ids = logits.max(dim=-1)

            # decoder_input_ids = torch.cat(max_ids, dim=-1)
            decoder_input_ids = torch.tensor(max_ids).to(src_input_ids.device)

            ids = max_ids.cpu().numpy().tolist()[0]

            if ids == vocab2id.get('<EOS>'):
                break
            result.append(ids)
        return result


if __name__ == '__main__':
    args = set_args()
    # tokenizer = BasicTokenizer()
    test_context = load_data(args.test_data_src)
    test_summary = load_data(args.test_data_tgt)
    vocab2id = json.load(open(args.vocab2id_path, 'r', encoding='utf8'))

    id2vocab = {id: vocab for vocab, id in vocab2id.items()}

    model = Model(vocab2id)
    model.load_state_dict(torch.load('./output_dir/epoch_19.bin', map_location='cpu'))

    if torch.cuda.is_available():
        model.cuda()

    final_context, final_summary, final_gen_summary = [], [], []
    for context, summary in tzip(test_context, test_summary):
        gen_summary = predict(model, vocab2id, context)
        gen_summary = ''.join([id2vocab[idx] for idx in gen_summary])
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
    
    
    

