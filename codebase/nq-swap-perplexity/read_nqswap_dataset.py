import os
import json
import gzip
import pandas as pd
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from constants import *
import ast

RAW_ROOT = os.path.join(CACHE_ROOT, 'NQ-swap/ml-knowledge-conflicts/datasets')

'''
Read Output dataset
'''

def read_logprob_output (type: str = 'Dev', dataset_name: str = 'original',
                         with_context: bool = True, teacher_dataset_name: str = None,
                         with_prob_seq: bool = False):
   root = os.path.join(CACHE_ROOT, 'NQ-swap/logprob_results')
   
   if not teacher_dataset_name:
      fname = f'MRQANaturalQuestions{type}-{dataset_name}-context={with_context}-inference.pickle'
   else:
      fname = f'MRQANaturalQuestions{type}-{dataset_name}-context={with_context}-teacherforcing={teacher_dataset_name}.pickle'

   df = pd.read_pickle(os.path.join(root, fname))

   df['answer_probs'] = df['answer_probs'].apply(lambda x: x.reshape(1) if x.shape == () else x)
   if teacher_dataset_name:
      df['teacher_answer_log_prob'] = df['answer_probs'].apply(lambda x: np.log(x).sum()/ len(x))
   else:
      df['inference_answer_log_prob'] = df['answer_probs'].apply(lambda x: np.log(x).sum()/ len(x))

   if not with_prob_seq:
      df.drop(columns = ['answer_probs'], inplace = True)
   
   return df

'''
Read and Load input dataset
'''

def load_input_dataset(type: str = 'Dev', dataset_name: str = 'original', with_context: bool = True, prefix_prompt = '',
                       teacher_forcing_dataset_name = None):
    df = read_raw_dataset(type=type, dataset_name=dataset_name)

    if with_context:
        df['input'] = prefix_prompt + 'source: ' + df['context'] + '\nquestion: ' + df['query'] +'\nanswer: '
    else:
        df['input'] = prefix_prompt + 'question: ' + df['query'] +'\nanswer: '
    
    if not teacher_forcing_dataset_name:
        return df
    elif teacher_forcing_dataset_name != dataset_name:
        teacher_df = read_raw_dataset(type = type, dataset_name = teacher_forcing_dataset_name)[['query', 'ground_truth']]
        teacher_df = teacher_df.explode('ground_truth').drop_duplicates().rename(columns = {'ground_truth': 'teacher_ans'})
        df = df.merge(teacher_df, on = 'query')
    else:
        df['teacher_ans'] = df['ground_truth'].copy()
        df = df.explode('teacher_ans')

    return df

def read_raw_dataset(type = 'Dev', dataset_name = 'original'):
   if dataset_name == 'original':
      filename = f'normalized/MRQANaturalQuestions{type}.jsonl.gz'
      examples = read_gzip_dataset(filename=filename)
      original_df = pd.DataFrame(examples)
      return original_df
   
   elif dataset_name in ['corpus-substitution', 'alias-substitution', 'popularity-substitution', 'type-swap-substitution']:
      substitution_df = load_substituion_df(type=type, substitution_method=dataset_name)
      return substitution_df
   else:
      raise ValueError(f"Invalid dataset name: {dataset_name}")

def load_substituion_df (type: str = 'Dev', substitution_method: str = 'corpus-substitution'):
    substitution_set = f'substitution-sets/MRQANaturalQuestions{type}-{substitution_method}.jsonl'
    examples = read_jsonl_dataset(root=RAW_ROOT, filename=substitution_set)
    return pd.DataFrame(examples)
 
def read_gzip_dataset (root: str = RAW_ROOT, filename: str = ''):
    path = os.path.join(root, filename)
    with gzip.open(path, 'r') as f:
        header = json.loads(f.readline())
        assert header["dataset"] in filename
        examples = [json.loads(line) for line in f.readlines()]
        for x in examples:
            x.update({'ground_truth': [ans['text'] for ans in x['gold_answers']]})
            x['context'] = x['context'].replace('<P>', '').replace('</P>', '')
        
        return examples
    
def read_jsonl_dataset (root: str = RAW_ROOT, filename: str = ''):
    path = os.path.join(root, filename)
    with open(path, 'r') as f:
        header = json.loads(f.readline())
        assert header["dataset"] in filename
        examples = [json.loads(line) for line in f.readlines()]
        for x in examples:
            x.update({'ground_truth': [ans['text'] for ans in x['gold_answers']]})
            x['context'] = x['context'].replace('<P>', '').replace('</P>', '')
        
        return examples
    
'''
Read and Save tokenized Dataset
'''

def save_tokenized_dataset (root = '/home/al2644/research/datasets/NQ-swap/tokenized_ground_truth_df',
                            suffix = None, type = 'Dev', dataset_name = 'original', with_context = True,
                            output_df = None):
   if dataset_name == 'original':
      filename = f'MRQANaturalQuestions{type}-context={with_context}'
   elif dataset_name in ['corpus-substitution', 'alias-substitution', 'popularity-substitution', 'type-swap-substitution']:
      filename = f'MRQANaturalQuestions{type}-{dataset_name}-context={with_context}'
   else:
      raise Exception('Uknown dataset name')
   
   if suffix is not None:
      filename += f'-{suffix}.csv'
   else:
      filename += '.csv'

   output_df.to_csv(os.path.join(root, filename), index = False)

def read_tokenized_dataset(root = '/home/al2644/research/datasets/NQ-swap/tokenized_ground_truth_df', suffix = None,
                           type = 'Dev', dataset_name = 'original', with_context = True):
   if dataset_name == 'original':
      filename = f'MRQANaturalQuestions{type}-context={with_context}'
   elif dataset_name in ['corpus-substitution', 'alias-substitution', 'popularity-substitution', 'type-swap-substitution']:
      filename = f'MRQANaturalQuestions{type}-{dataset_name}-context={with_context}'
   else:
      raise Exception('Uknown dataset name')
   
   if suffix is not None:
      filename += f'-{suffix}.csv'
   else:
      filename += '.csv'
    
   return pd.read_csv(os.path.join(root, filename))

'''
Read from and Save to pickle
'''

def save_to_pickle (root: str = '', filename: str = '', data_obj = None):
  with open(os.path.join(root, filename), 'wb') as f:
    pickle.dump(data_obj, f)

def read_from_pickle (root: str = '', filename: str = ''):
  with open(os.path.join(root, filename), 'rb') as f:
    return pickle.load(f)
   