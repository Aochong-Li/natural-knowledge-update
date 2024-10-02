import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import spacy
import tqdm

from nq_swap_inference import *
from read_nqswap_dataset import *

def filter_tokenize_ground_truth (input_df: pd.DataFrame, mdl_name: str = 'meta-llama/Llama-2-7b-chat-hf'):
    tokenizer = AutoTokenizer.from_pretrained(mdl_name, padding_side = 'left')
    nlp = spacy.load("en_core_web_sm")
    
    filtered_tokens, filtered_ids = [], []

    for i in tqdm(range(len(input_df))):
        ground_truth = ' '.join(input_df['ground_truth'][i]) # can be multiple ground truth
        # step 1 filter out filler words from ground_truth
        filtered_ground_truth = [token.text for token in nlp(ground_truth) if not (token.is_stop or token.is_punct)]
        # step 2 tokenize the filtered ground truth
        filtered_ground_truth_id = tokenizer(' '.join(filtered_ground_truth), add_special_tokens=False)['input_ids']

        filtered_tokens.append(filtered_ground_truth)
        filtered_ids.append(filtered_ground_truth_id)
    
    input_df['ground_truth_tokens'] =  filtered_tokens
    input_df['ground_truth_token_ids'] = filtered_ids
    
    return input_df

if __name__ == "__main__":
    dataset_type = 'Dev'
    dataset_name =  'original' # 'original' # 'corpus-substitution'
    with_context = False
    prefix_prompt = '''Instruction: answer the below question in an accurate, concise way.\n'''

    input_df = load_dataset(type = dataset_type, dataset_name = dataset_name, with_context = with_context, prefix_prompt = prefix_prompt)
    output_df = filter_tokenize_ground_truth(input_df = input_df)

    save_tokenized_dataset(root = '/home/al2644/research/datasets/NQ-swap/tokenized_ground_truth_df', suffix='tokenized',
                 type = dataset_type, dataset_name = dataset_name, with_context = with_context, output_df = output_df)
