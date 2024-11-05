from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import os
import sys
import json
import pickle
import logging
import numpy as np
import torch
import torch.nn.functional as F
from read_nqswap_dataset import read_raw_dataset, load_input_dataset
import pandas as pd
from tqdm import tqdm
from constants import *

import argparse

BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
SPACE_TOKEN_ID = 29871
ROOT = os.path.join(CACHE_ROOT, 'NQ-swap/logprob_results')

def save_inference_to_pickle (root = ROOT, filename = '', batch_prompts = [], decoded_ouput = [], ans_token_ids = None,  probs = None):
    print('Cachcing results')
    if os.path.isfile(os.path.join(root, filename)):
        df = pd.read_pickle(os.path.join(root, filename))
    else:
        df = pd.DataFrame(columns = ['input', 'decoded_answer', 'answer_probs'])
    new_data = [df]
    
    for i in range(len(batch_prompts)):
        prob_mat = probs[i, :len(ans_token_ids[i]), :]
        prob_seq = prob_mat[torch.arange(prob_mat.size(0)), ans_token_ids[i]].numpy()

        data_obj = {'input': batch_prompts[i],
                    'decoded_answer': decoded_ouput[i].replace(batch_prompts[i], ''),
                    'answer_probs': prob_seq}
        new_data.append(pd.DataFrame.from_dict(data_obj, orient = 'index').T)
    
    pd.concat(new_data).reset_index(drop=True).to_pickle(os.path.join(root, filename), protocol=pickle.HIGHEST_PROTOCOL)

def save_teacher_forcing_inference_to_pickle (root = ROOT, filename = '', batch_prompts = [], teacher_ans = [],
                                              teacher_ans_len = None, labels = None, probs = None):
    print('Cachcing results')
    if os.path.isfile(os.path.join(root, filename)):
        df = pd.read_pickle(os.path.join(root, filename))
    else:
        df = pd.DataFrame(columns = ['input', 'teacher_answer', 'answer_probs'])
    new_data = [df]
    for i in range(len(batch_prompts)):
        prob_mat = probs[i][: teacher_ans_len[i]]
        prob_seq = torch.gather(prob_mat, 1, labels[i][:teacher_ans_len[i]].unsqueeze(1)).squeeze().numpy()

        data_obj = {'input': batch_prompts[i],
                    'teacher_answer': teacher_ans[i],
                    'answer_probs': prob_seq}
        new_data.append(pd.DataFrame.from_dict(data_obj, orient='index').T)

    pd.concat(new_data).reset_index(drop = True).to_pickle(os.path.join(root, filename), protocol=pickle.HIGHEST_PROTOCOL)

def model_inference (model, tokenizer, input_df: pd.DataFrame, output_fname: str = '',
                     max_new_tokens: int = 256, batch_size: int = 64,last_batch: int = 0,
                     temperature: float = 0.0, top_p: float = 1.0, top_k: int = 50):
    '''WAWRNING: this code assumes only greedy decoding'''
    model.eval()

    unique_inputs = input_df['input'].unique()

    for i in tqdm(range(last_batch, len(unique_inputs), batch_size)):
        '''preparing the batch'''
        print(f'preparing batch {i} to {i + batch_size}')
        batch_prompts = list(unique_inputs[i : i + batch_size])
        batch_inputs = tokenizer(
            batch_prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=model.config.max_position_embeddings
            )
        
        if torch.cuda.is_available():
            batch_inputs = {k: v.cuda() for k,v in batch_inputs.items()}

        '''forward pass of the batch'''
        with torch.inference_mode():
            outputs = model.generate(**batch_inputs,
                                     max_new_tokens=max_new_tokens,
                                     do_sample=False,
                                     return_dict_in_generate = True,
                                     output_logits = True
                                     )
                   
            '''get probabilities'''
            logits = torch.stack(outputs.logits).to('cpu')
            probs = F.softmax(logits, dim=-1).permute(1, 0, 2) # dimension of batch_size, time, vocab_size

            tokenids = logits.argmax(axis = -1).T # output token ids dimension of batch_size, time
            EOS_mask = (tokenids == EOS_TOKEN_ID) # wher EOS is in the output
            mask = torch.where(EOS_mask.any(dim = 1), EOS_mask.float().argmax(dim = 1), EOS_mask.size(1)).tolist()

            '''extend decoded outputs'''
            sequences = outputs.sequences.cpu()
            decoded_text = tokenizer.batch_decode(sequences, skip_special_tokens=True)

            ans_token_ids_batch = [tokenids[idx][: mask[idx]].tolist() for idx in range(len(tokenids))]
            
            '''cache the results'''
            save_inference_to_pickle(filename=output_fname, batch_prompts = batch_prompts,
                           decoded_ouput = decoded_text, ans_token_ids=ans_token_ids_batch, probs=probs)

        del batch_inputs, outputs
        torch.cuda.empty_cache()

def model_teacher_forcing_inference (model, tokenizer, input_df: pd.DataFrame, output_fname: str = '',
                                     max_new_tokens: int = 256, batch_size: int = 64,last_batch: int = 0,
                                     temperature: float = 0.0, top_p: float = 1.0, top_k: int = 50):
    '''WAWRNING: this code assumes only greedy decoding'''
    model.eval()
    unique_input_ans = input_df[['input', 'teacher_ans']].drop_duplicates()

    for i in tqdm(range(last_batch, len(unique_input_ans), batch_size)):
        '''preparing the batch'''
        print(f'preparing batch {i} to {i + batch_size}')
        batch_prompts = list(unique_input_ans['input'][i : i + batch_size])
        batch_teacher_ans = list(unique_input_ans['teacher_ans'][i : i + batch_size])

        # '''Debugging START'''
        # if EXAMPLE not in batch_prompts:
        #     continue
        # else:
        #     import pdb; pdb.set_trace()
        # '''Debugging END'''

        '''tokenized the batch'''
        tokenizer.padding_side = 'left'
        batch_tokenized_prompts = tokenizer(
            batch_prompts,
            return_tensors = 'pt',
            padding = True,
            truncation = True,
            max_length = model.config.max_position_embeddings
        )
        prompt_len = batch_tokenized_prompts['input_ids'].size(1)

        tokenizer.padding_side = 'right'
        batch_tokenized_ans = tokenizer(
            batch_teacher_ans,
            return_tensors = 'pt',
            padding = True,
            truncation = True,
            max_length = model.config.max_position_embeddings
        )
        '''remove BOS token of answer'''
        batch_tokenized_ans['input_ids'] = batch_tokenized_ans['input_ids'][:, 1: ]
        batch_tokenized_ans['attention_mask'] = batch_tokenized_ans['attention_mask'][:, 1: ]

        teacher_ans_len = batch_tokenized_ans['attention_mask'].sum(axis = 1)

        if torch.cuda.is_available():
            batch_inputs = {k: torch.concat([batch_tokenized_prompts[k], batch_tokenized_ans[k]], axis = 1) for k in batch_tokenized_prompts.keys()}
            batch_inputs = {k: v.cuda() for k,v in batch_inputs.items()}

            labels = batch_inputs['input_ids'].clone()
            labels[labels == 0] = -100
            labels = labels.cuda()

        '''forward pass of the batch'''
        with torch.inference_mode():
            outputs = model(
                input_ids = batch_inputs['input_ids'],
                attention_mask = batch_inputs['attention_mask'],
                labels = labels,
                output_attentions = True
                )
            # start, end = 0, input_ids.size(1)
            # start, end = 20, 25
            # position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
            # position_ids = position_ids.unsqueeze(0).expand(input_ids.size(0), -1)

            # outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels=label, output_attentions = True, position_ids=position_ids)

            # outputs = model(input_ids = input_ids[:, start: end], attention_mask = attention_mask[:, start: end], labels=label[:, start: end], output_attentions = True)
            # outputs.logits

            logits = outputs.logits.to('cpu')
            ans_logits = logits[:, prompt_len - 1: ,:]

            '''get probabilities'''
            probs = F.softmax(ans_logits, dim=-1)# dimension of batch_size, time, vocab_size

            '''cache results'''
            # DEBUGGING
            save_teacher_forcing_inference_to_pickle(filename=output_fname, batch_prompts=batch_prompts, teacher_ans=batch_teacher_ans,
                                                     teacher_ans_len=teacher_ans_len, labels=labels[:, prompt_len: ].to('cpu'),probs=probs)
        
        del batch_inputs, labels, outputs
        torch.cuda.empty_cache()

def load_model(mdl_name: str, float16qtz: bool = True):
    LOGIN_TOKEN = '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'
    login(token=LOGIN_TOKEN)

    tokenizer = AutoTokenizer.from_pretrained(mdl_name, padding_side = 'left')
    tokenizer.pad_token = "[PAD]"

    model = AutoModelForCausalLM.from_pretrained(mdl_name, torch_dtype=torch.float16 if float16qtz else None)

    if torch.cuda.device_count() > 1:
        print('spread data across devices')
        model = torch.nn.DataParallel(model)
    
    elif torch.cuda.is_available():
        model.to('cuda')
    
    return tokenizer, model

if __name__ == "__main__":
    '''Parsing Arguments'''
    parser = argparse.ArgumentParser(description = 'Parsing arguments for nq_swap_inference.py')
    parser.add_argument('--dataset_name', type=str, help='original, corpus-substitution')
    parser.add_argument('--with_context', type=lambda x: (str(x).lower() == 'true'), help='if context is added')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--teacher_dataset_name', type=str, help='dataset for teaching forcing answers')
    args = parser.parse_args()

    '''Set Hyperparameters'''
    mdl_name = "meta-llama/Llama-2-7b-chat-hf"
    temperature, top_p, top_k = 0.0, 1.0, 50
    max_new_tokens, batch_size = 128, args.batch_size

    dataset_type = 'Dev'
    dataset_name =  args.dataset_name
    with_context = args.with_context
    tokenizer, model = load_model(mdl_name = mdl_name, float16qtz = True)
    df = load_input_dataset(type = dataset_type,
                            dataset_name = dataset_name,
                            with_context = with_context,
                            prefix_prompt = PREFIX_PROMPT,
                            teacher_forcing_dataset_name = args.teacher_dataset_name)
    
    if not args.teacher_dataset_name:
        output_fname = f'MRQANaturalQuestions{dataset_type}-{dataset_name}-context={with_context}-inference.pickle'
    
        model_inference(model = model,
                        tokenizer = tokenizer,
                        input_df = df,
                        output_fname = output_fname,
                        max_new_tokens = max_new_tokens,
                        batch_size = batch_size,
                        temperature = temperature,
                        top_p = top_p,
                        top_k = top_k)
    else:
        output_fname = f'MRQANaturalQuestions{dataset_type}-{dataset_name}-context={with_context}-teacherforcing={args.teacher_dataset_name}.pickle'
        model_teacher_forcing_inference(model=model,
                                        tokenizer=tokenizer,
                                        input_df=df,
                                        output_fname=output_fname,
                                        max_new_tokens=max_new_tokens,
                                        batch_size=batch_size,
                                        temperature=temperature,
                                        top_p=top_p,
                                        top_k=top_k)
