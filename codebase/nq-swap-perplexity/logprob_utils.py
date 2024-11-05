import os
import json
import gzip
import pandas as pd
import ast
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from read_nqswap_dataset import *

ROOT = '/home/al2644/research/datasets/NQ-swap/logprob_results'
INPUT = ["Instruction: answer the below question in an accurate, concise way.\nquestion: who has been chosen as the brand ambassador of the campaign ' beti bachao-beti padhao\nanswer: "]

def compute_ans_logprob (root = ROOT, filename = '', input_df = None, original_df = None, tokenizer = None):
   '''
   we want to get the probabilities of three things
   1. log prob of the entire answer
   2. if the answer match with the ground truth answer, what is the prob of the ground truth

   3. if it is a substitution, what is the log prob of the original answer
   '''
   if tokenizer is None:
      tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", padding_side = 'left')
   
   itemidx = 0
   output_df_list = []

   with open(os.path.join(root, filename), 'rb') as f:
      while True:
        print(f'processing {itemidx}')
        try:
            output = pickle.load(f)
            
            '''
            Debugging
            TODO: Comment out when not needed
            '''
            # if output['input'] in INPUT:
            #    import pdb; pdb.set_trace()
            # else:
            #    continue

            output_ids = tokenizer.convert_tokens_to_ids(output['ouput_tokens']) # tokenizer.convert_ids_to_tokens(ground_truth_token_ids)
            length, dim = output['output_probs'].shape
            
            # log prob of the output
            output_mask = np.eye(dim)[np.array(output_ids)].astype(int)
            output_logprob = np.log(np.sum(output['output_probs'] * output_mask, axis = -1)).sum() / length
            
            # log prob of anwers in output that match ground truth
            ground_truth_token_ids = input_df[input_df['input'] == output['input']]['ground_truth_token_ids'].iloc[0]
            ground_truth_token_ids = ast.literal_eval(ground_truth_token_ids)
            ground_truth_token_mask = np.zeros((length, dim), dtype=int)
            ground_truth_token_mask[ : ,ground_truth_token_ids] = 1

            overlapping_token_id_mask = output_mask & ground_truth_token_mask
            overlapping_tokens_pos = np.sum(overlapping_token_id_mask, axis = 1)
            overlapping_tokens_prob = np.sum(output['output_probs'] * overlapping_token_id_mask, axis = -1)
            overlapping_tokens_logporb = np.log(overlapping_tokens_prob[overlapping_tokens_pos == 1])
            ans_logprob = np.sum(overlapping_tokens_logporb) / np.sum(overlapping_tokens_pos) \
                          if np.sum(overlapping_tokens_pos) > 0 else np.nan

            # log prob of ground truth
            # ground_truth_seq_mask = np.eye(dim)[np.array(ground_truth_token_ids)].astype(int)[:]

            # drop prob dist sequences
            output.pop('output_probs')
            output['output_logprob'] = output_logprob
            output['ans_logprob'] = ans_logprob

            output_df_list.append(pd.DataFrame.from_dict(output, orient='index').T) 

            itemidx +=1
        except EOFError:
           break
        except Exception as e:
           print(f'An error occured when reading {os.path.join(root, filename)}')
           print(f'Error Message: {e}')
           break
   
   return pd.concat(output_df_list).reset_index(drop=True)

if __name__ == "__main__":
    dataset_type = 'Dev'
    dataset_name =  'original' # 'original' # 'corpus-substitution'
    with_context = False

    tokenized_df = read_tokenized_dataset(root = '/home/al2644/research/datasets/NQ-swap/tokenized_ground_truth_df', suffix='tokenized',
                                          type = dataset_type, dataset_name = dataset_name, with_context=with_context)
    
    logprob_df = compute_ans_logprob(root = ROOT, filename = f'MRQANaturalQuestions{dataset_type}-{dataset_name}-context={with_context}.pickle',
                                     input_df = tokenized_df)
    
   #  logprob_df.to_csv(os.path.join(ROOT, f'MRQANaturalQuestions{dataset_type}-{dataset_name}-context={with_context}_logprob.csv'), index = False)
