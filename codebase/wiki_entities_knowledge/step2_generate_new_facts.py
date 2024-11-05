from cache_batch_input import *
from codebase.wiki_entities_knowledge.postprocess_outputs import *
from prompts import *
import openaiAPI
import os
import json

def run_model (df, input_prompt_template, system_prompt, template_properties,
              model, temperature, max_tokens, batch_size, input_filepath, batch_log_filepath,
              cache_filepath, mode):
    '''prepare batch input'''

    if not os.path.exists(batch_log_filepath) or mode == 'stream_generate':
        '''prepare batch input'''
        prepare_batch_input(df = df, input_prompt_template = input_prompt_template,
                            template_properties = template_properties,
                            system_prompt = system_prompt, model = model,
                            temperature = temperature, max_tokens = max_tokens,
                            input_filepath = input_filepath)
        
        print(f'Batch input prepared and stored at {input_filepath}')

        '''generate'''

        if mode == 'stream_generate':
            if os.path.exists(cache_filepath):
                print(f'Results are stored from {cache_filepath}')
            else:
                openaiAPI.stream_generate_response(input_filepath=input_filepath, cache_filepath=cache_filepath)
                print(f'Results are generated and stored at {cache_filepath}')
            
        else:   
            openaiAPI.minibatch_stream_generate_response(input_filepath=input_filepath, batch_log_filepath=batch_log_filepath, batch_size = batch_size)
            print(f'Results are generated and check {batch_log_filepath}')

def generate_new_facts (df, entity_col = 'title', fact_col = 'step1v_output',
                        model = 'gpt-4o', temperature = 1.0, max_tokens = 512, batch_size = 20,
                        input_filepath = None, batch_log_filepath = None, cache_filepath = None, mode = 'batch_stream'):
    
    input_prompt_template = STEP2_INPUT_PROMPT_TEMPLATE
    system_prompt = SYSTEM_PROMPT

    template_properties = {'entity': entity_col, 'fact': fact_col}

    run_model(df = df, input_prompt_template = input_prompt_template, system_prompt = system_prompt,
            template_properties = template_properties, model = model, temperature = temperature,
            max_tokens = max_tokens, batch_size = batch_size, input_filepath = input_filepath,
            batch_log_filepath = batch_log_filepath, cache_filepath = cache_filepath, mode = mode)


def retrieve_outputs (batch_log_filepath, cache_filepath):
    if os.path.exists(cache_filepath):
        print(f'Results are retrieved from {cache_filepath}')
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
        print(f'Facts retrieved and stored at {cache_filepath}')

    return output_df
