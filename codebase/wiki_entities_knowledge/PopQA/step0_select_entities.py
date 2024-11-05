import os
import sys
import json
sys.path.append('/home/al2644/research/codebase/wiki_entities_knowledge')

from cache_batch_input import *
from codebase.wiki_entities_knowledge.postprocess_outputs import *
from prompts import *
import openaiAPI

def select_entities (df, entity_col='title', summary_col='truncated_summary', model='gpt-4o-mini',
                           temperature=0.5, max_tokens = 512, batch_size = 20,
                           input_filepath = None, batch_log_filepath = None, cache_filepath = None, mode = 'batch_stream'):
    
    input_prompt_template = STEP0_INPUT_PROMPT_TEMPLATE
    system_prompt = SYSTEM_PROMPT

    template_properties = {'entity': entity_col, 'summary': summary_col}

    if not os.path.exists(batch_log_filepath) or mode == 'stream_generate':
        '''prepare batch input'''
        prepare_batch_input(df = df, input_prompt_template = input_prompt_template,
                            template_properties = template_properties,
                            system_prompt = system_prompt, model = model,
                            temperature = temperature, max_tokens = max_tokens,
                            input_filepath = input_filepath)
        
        print(f'Batch input prepared and stored at {input_filepath}')

        '''select entities'''
        if mode == 'stream_generate':
            if os.path.exists(cache_filepath):
                print(f'Facts retrieved from {cache_filepath}')
                output_df = pd.read_pickle(cache_filepath)
            else:
                output_df =openaiAPI.stream_generate_response(input_filepath=input_filepath, cache_filepath=cache_filepath)
                print(f'Results are generated and stored at {cache_filepath}')
            
            return output_df
        else:   
            openaiAPI.minibatch_stream_generate_response(input_filepath=input_filepath, batch_log_filepath=batch_log_filepath, batch_size = batch_size)
            print(f'Results are generated and check {batch_log_filepath}')

def retrieve_outputs (batch_log_filepath, cache_filepath):
    if os.path.exists(cache_filepath):
        print(f'Facts retrieved from {cache_filepath}')
        output_df = pd.read_pickle(cache_filepath)
    else:
        with open(batch_log_filepath) as f:
            batch_logs = json.load(f)

        output_dict = {}
        for idx, batch_log_id in batch_logs.items():
            status = openaiAPI.check_batch_status(batch_log_id)
            if status == 'completed':
                output_file_id = openaiAPI.retrieve_batch_output_file_id(batch_log_id)
                output_dict[idx] = output_file_id
            else:
                print(f'Batch {batch_log_id} at {idx} failed')

        output_df = openaiAPI.minibatch_retrieve_response(output_dict_filepath=None, output_dict=output_dict)
        output_df.to_pickle(cache_filepath)
        print(f'Results are retrieved and stored at {cache_filepath}')

        return output_df
    
def postprocess_outputs (input_df,output_df):
    output_df.rename(columns = {'response': 'step0_response'}, inplace=True)
    df  = input_df.merge(output_df, left_index=True, right_index=True)

    df = df[df['step0_response'].str.contains('Output')]
    df['step0_output'] = df['step0_response'].apply(lambda x: True if '[True]' in x.split('Output')[1] else False)

    return df.drop(columns=['step0_response'])