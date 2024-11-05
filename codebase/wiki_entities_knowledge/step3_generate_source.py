from cache_batch_input import *
from codebase.wiki_entities_knowledge.postprocess_outputs import *
from prompts import *
import openaiAPI
from openaiAPI import *
import os
import numpy as np

news_outlets = ['ABC News', 'BBC News', 'CNN', 'Fox News', 'The New York Times', 'The Wall Street Journal']

def generate_explicit_news (df, entity_col = 'title', fact_col = 'step2_new_fact', old_fact_col = 'step1v_output', reason_col = 'step2_reasoning',
                        model = 'gpt-4o-mini', temperature = 1.0, max_tokens = 2048, batch_size = 10,
                        input_filepath = None, batch_log_filepath = None, cache_filepath = None, mode = 'batch_stream'):
       input_prompt_template = STEP3_EXPLICIT_INPUT_PROMPT_TEMPLATE
       system_prompt = STEP3_SYSTEM_PROMPT

       np.random.seed(42)  # Set random seed for reproducibility
       sources = np.random.choice(news_outlets, size = len(df), replace = True)
       df['source'] = sources
       
       template_properties = {
           'source': 'source',
           'entity': entity_col,
           'new_fact': fact_col,
           'old_fact': old_fact_col,
           'reason': reason_col
           }
       
       run_model(df = df, input_prompt_template = input_prompt_template, system_prompt = system_prompt,
            template_properties = template_properties, model = model, temperature = temperature,
            max_tokens = max_tokens, batch_size = batch_size, input_filepath = input_filepath,
            batch_log_filepath = batch_log_filepath, cache_filepath = cache_filepath, mode = mode)
       
def generate_news (df, entity_col = 'title',model = 'gpt-4o-mini', temperature = 1.0, max_tokens = 2048, batch_size = 10,
                   input_filepath = None, batch_log_filepath = None, cache_filepath = None, mode = 'batch_stream'):
       input_prompt_template = STEP3_NEWS_INPUT_PROMPT_TEMPLATE
       system_prompt = STEP3_SYSTEM_PROMPT

       np.random.seed(42)  # Set random seed for reproducibility
       sources = np.random.choice(news_outlets, size = len(df), replace = True)
       df['source'] = sources
       
       template_properties = {
           'source': 'source',
           'entity': entity_col
           }
       
       run_model(df = df, input_prompt_template = input_prompt_template, system_prompt = system_prompt,
            template_properties = template_properties, model = model, temperature = temperature,
            max_tokens = max_tokens, batch_size = batch_size, input_filepath = input_filepath,
            batch_log_filepath = batch_log_filepath, cache_filepath = cache_filepath, mode = mode)

def generate_implicit_news (df, entity_col = 'title', fact_col = 'step2_new_fact', article_col = 'step3_article',
                            model = 'gpt-4o-mini', temperature = 1.0, max_tokens = 2048, batch_size = 10,
                            input_filepath = None, batch_log_filepath = None, cache_filepath = None, mode = 'batch_stream'):
       input_prompt_template = STEP3_IMPLICIT_INPUT_PROMPT_TEMPLATE
       system_prompt = STEP3_SYSTEM_PROMPT
       
       template_properties = {
           'entity': entity_col,
           'new_fact': fact_col,
           'article': article_col
           }
       
       run_model(df = df, input_prompt_template = input_prompt_template, system_prompt = system_prompt,
            template_properties = template_properties, model = model, temperature = temperature,
            max_tokens = max_tokens, batch_size = batch_size, input_filepath = input_filepath,
            batch_log_filepath = batch_log_filepath, cache_filepath = cache_filepath, mode = mode)
       
def generate_articles_deprecated (df, entity_col='title', fact_col='facts', counterfact_col='counterfact', reason_col='reasoning',
                       model='gpt-4o', temperature=1.0, max_tokens=2048, batch_size = 5, input_filepath = None,
                       output_filepath = None, cache_filepath = None, mode = 'batch_stream'):
    
    input_prompt_template = STEP3_INPUT_PROMPT_TEMPLATE
    system_prompt = STEP3_SYSTEM_PROMPT

    template_properties = {'entity': entity_col, 'fact': fact_col, 'counterfact': counterfact_col, 'reason': reason_col}

    if not os.path.exists(output_filepath) or mode == 'stream_generate':
        '''prepare batch input'''
        prepare_batch_input(df = df, input_prompt_template = input_prompt_template, system_prompt=system_prompt,
                            template_properties = template_properties, model = model,
                            temperature = temperature, max_tokens = max_tokens,
                            input_filepath = input_filepath)
        
        print(f'Batch input prepared and stored at {input_filepath}')

        '''generate counterfacts'''
        if mode == 'stream_generate':
            if os.path.exists(cache_filepath):
                print(f'Facts retrieved from {cache_filepath}')
                output_df = pd.read_csv(cache_filepath)
            else:
                output_df =openaiAPI.stream_generate_response(input_filepath=input_filepath, cache_filepath=cache_filepath)
                print(f'Facts generated and stored at {cache_filepath}')
            
            return output_df
        else:   
            openaiAPI.minibatch_stream_generate_response(input_filepath=input_filepath, output_filepath=output_filepath, batch_size = batch_size)
            print(f'Facts generated and stored at {output_filepath}')

    '''retrieve outputs'''
    output_df = openaiAPI.minibatch_retrieve_response(output_filepath)
    print(f'Facts retrieved from {output_filepath}')

    return output_df
