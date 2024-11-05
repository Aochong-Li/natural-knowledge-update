import pandas as pd
import os
import sys
import json
from datasets import load_dataset
import random

from codebase.wikidata import get_wiki_data
from tqdm import tqdm

SHARE_ROOT = '/share/goyal/lio/knowledge_update/wikipedia/PopQA'
ROOT_PATH = '/home/al2644/research/datasets/adaptive-retrieval/data/results'


def add_wiki_summary (df, col = 'title'):
    output_df = df.copy()
    summaries = []

    for i in tqdm(range(len(df[col]))):
        title = df[col].iloc[i]
        page_summary = get_wiki_data.get_page_summary(title)
        if page_summary == "":
            page_summary = df['text'].iloc[i][:600]
        summaries.append(page_summary)
    
    output_df['wiki_summary'] = summaries

    return output_df

def load_input_df (filename: str) -> pd.DataFrame:
    input_df = pd.read_csv(os.path.join(ROOT_PATH, filename)).drop(columns = 'Unnamed: 0')
    columns = ['id', 's_wiki_title']
    input_df = input_df[input_df['is_correct']]

    return input_df[columns].rename(columns = {'id': 'popqa_id', 's_wiki_title': 'title'})

def prepare_step0_input (input_df: pd.DataFrame) -> pd.DataFrame:
    dataset_name = "wikimedia/wikipedia"
    version = "20231101.en"
    cache_dir = "/share/goyal/lio/dataset/huggingface"
    dataset = load_dataset(dataset_name, version, cache_dir=cache_dir)['train']
    import pdb; pdb.set_trace()

    popqa_titles = set(input_df['title'])

    def batch_filter (batch):
        return [title in popqa_titles for title in batch['title']]

    wiki_dataset = dataset.filter(batch_filter, batched = True, batch_size = 5000, num_proc = 32)
    wiki_df = wiki_dataset.to_pandas()

    return wiki_df

if __name__ == '__main__':
    filename = 'model=meta-llama_Llama-2-7b-chat-hf-input=llama2_7b-method=vanilla-shots=15-n=14267.csv'
    input_df = load_input_df(filename)
    res_df = prepare_step0_input(input_df)

    res_df = add_wiki_summary(res_df)

    output_file_path = os.path.join(SHARE_ROOT, 'step0_input.pickle')
    res_df.to_pickle(output_file_path)