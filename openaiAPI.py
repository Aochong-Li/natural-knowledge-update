import openai
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import os
import time

ORG_ID = ''
PROJECT_ID = ''
OPENAI_API_KEY = ''

client = OpenAI(
    api_key=OPENAI_API_KEY,
    organization=ORG_ID,
    project=PROJECT_ID,
)

'''generate response'''

def generate_response(input_prompt: str, system_prompt: str = 'You are a helpful assistant', model: str = 'gpt-4o',
                      temperature: float = 0.0, max_tokens: int = 1024, top_p: float = 1.0, frequency_penalty: float = 0.0,
                      presence_penalty: float = 0.0, stop: list[str] = None ):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
    )
    return response.choices[0].message.content

def stream_generate_response(input_filepath: str, cache_filepath: str, failed_start: int = None, failed_end: int = None):

    output_df = pd.DataFrame(columns = ['response'])

    with open(input_filepath, 'r') as f:
        batch_input = [json.loads(line) for line in f]
        if failed_end is not None:
            batch_input = batch_input[: failed_end]
        if failed_start is not None:
            batch_input = batch_input[failed_start:]

    for i in tqdm(range(len(batch_input))):
        input_object = batch_input[i]
        id = int(input_object['custom_id'].replace('idx_', ''))
        input_prompt = input_object['body']['messages'][1]['content']
        system_prompt = input_object['body']['messages'][0]['content']
        model = input_object['body']['model']
        temperature = input_object['body']['temperature']
        max_tokens = input_object['body']['max_tokens']
        top_p = input_object['body']['top_p']
        frequency_penalty = input_object['body']['frequency_penalty']
        presence_penalty = input_object['body']['presence_penalty']
        stop = input_object['body']['stop']

        response = generate_response(input_prompt = input_prompt, system_prompt = system_prompt, model = model,
                                     temperature = temperature, max_tokens = max_tokens, top_p = top_p,
                                     frequency_penalty = frequency_penalty, presence_penalty = presence_penalty, stop = stop)
        
        output_df.loc[id] = [response]
        
        output_df.to_pickle(cache_filepath)

def minibatch_stream_retry (batch_log_filepath: str, batch_rate_limit: int = None):
    failed_batch_logs = {}
    retry_batch_logs = {}

    with open(batch_log_filepath, 'r') as f:
        batch_logs = json.load(f)
    
    for batch_idx, batch_log_id in batch_logs.items():
        status = check_batch_status(batch_log_id)
        if status == 'failed':
            failed_batch_logs[batch_idx] = batch_log_id
    
    for batch_idx, batch_log_id in failed_batch_logs.items():
        print(f'Retrying batch {batch_idx}')
        
        batch_log = client.batches.retrieve(batch_log_id)
        batch_input_file_id = batch_log.input_file_id
        completion_window = batch_log.completion_window

        batch_log = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata={
            "description": f"minibatch_{batch_idx}"
            }
        )
        print(f'batch {batch_log.id} is created')

        retry_batch_logs[batch_idx] = batch_log.id

        if batch_rate_limit is not None and len(retry_batch_logs) % batch_rate_limit == 0:
            time.sleep(30)
        
        batch_logs.update(retry_batch_logs)

        with open(batch_log_filepath, 'w') as f:
            json.dump(batch_logs, f)

def minibatch_stream_generate_response(input_filepath: str,
                                       batch_log_filepath: str = None,
                                       minibatch_filepath: str = '/home/al2644/research/openai_batch_io/minibatchinput.jsonl',
                                       batch_size: int = 10,
                                       completion_window: str = '24h',
                                       failed_batch_start: int = None,
                                       failed_batch_end: int = None,
                                       batch_rate_limit: int = None):
    batch_logs = {}

    with open(input_filepath, 'r') as f:
        batch_input = [json.loads(line) for line in f]
        if failed_batch_start is not None and failed_batch_end is not None:
            batch_input = batch_input[failed_batch_start: failed_batch_end]

    while len(batch_logs) * batch_size < len(batch_input):
        batch_idx = batch_size * len(batch_logs)

        with open(minibatch_filepath, 'w') as f:
            for item in batch_input[batch_idx : batch_idx + batch_size]:
                f.write(json.dumps(item) + '\n')
        
        # uplaod batch input files
        batch_input_file = client.files.create(
            file=open(minibatch_filepath, "rb"),
            purpose="batch"
        )
        
        # create batch
        batch_input_file_id = batch_input_file.id

        batch_log = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata={
            "description": f"minibatch_{batch_idx}"
            }
        )
        print(f'batch {batch_log.id} is created')

        batch_logs[batch_idx] = batch_log.id


        if batch_rate_limit is not None and len(batch_logs) % batch_rate_limit == 0:
            time.sleep(30)

        with open(batch_log_filepath, 'w') as f:
            json.dump(batch_logs, f)

def minibatch_stream_generate_response_deprecated(input_filepath: str,
                                       output_filepath: str,
                                       minibatch_filepath: str = '/home/al2644/research/openai_batch_io/minibatchinput.jsonl',
                                       batch_size: int = 10,
                                       completion_window: str = '24h',
                                       failed_batch_start: int = None,
                                       failed_batch_end: int = None):
    import time
    outputs = {}
    with open(input_filepath, 'r') as f:
        batch_input = [json.loads(line) for line in f]
        if failed_batch_start is not None and failed_batch_end is not None:
            batch_input = batch_input[failed_batch_start: failed_batch_end]

    while len(outputs) * batch_size < len(batch_input):
        batch_idx = batch_size * len(outputs)

        with open(minibatch_filepath, 'w') as f:
            for item in batch_input[batch_idx : batch_idx + batch_size]:
                f.write(json.dumps(item) + '\n')
        
        # uplaod batch input files
        batch_input_file = client.files.create(
            file=open(minibatch_filepath, "rb"),
            purpose="batch"
        )
        
        # create batch
        batch_input_file_id = batch_input_file.id

        batch_log = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata={
            "description": f"minibatch_{batch_idx}"
            }
        )
        print(f'batch {batch_log.id} is created')
        
        while client.batches.retrieve(batch_log.id).status != 'completed':
            print(f'still working on batch {batch_log.id}')
            time.sleep(15)
            
            if client.batches.retrieve(batch_log.id).status == 'failed':
                print(f'Job failed at batch {batch_idx}')
                break
        
        print(f'batch {batch_log.id}, input {batch_idx} to {batch_idx + batch_size} are completed')   
        output_file_id = client.batches.retrieve(batch_log.id).output_file_id
        outputs[batch_idx] = output_file_id
        print(f'output file: {output_file_id}')
        
        with open(output_filepath, 'w') as f:
            json.dump(outputs, f)

def minibatch_retrieve_response(output_dict_filepath: str = None, output_dict: dict = None):
    
    if output_dict_filepath is not None:
        with open(output_dict_filepath, 'r') as f:
            output_dict = json.load(f)
    
    model_outputs = {}
    for _, output_file_id in output_dict.items():
        file_response = client.files.content(output_file_id)
        print(f'Retrieving output {output_file_id}')
        
        text_responses = file_response.text.split('\n')[:-1]
        json_responses = [json.loads(x) for x in text_responses]
        
        for output in json_responses:
            custom_id = int(output['custom_id'].replace('idx_', ''))
            content = output['response']['body']['choices'][0]['message']['content']
            model_outputs[custom_id] = content
    
    return pd.DataFrame.from_dict(model_outputs, orient='index', columns = ['response'])
        
def batch_generate_response(batch_json_filepath: str, completion_window: str = '24h', description: str = None):
    batch_input_file = client.files.create(file = open(batch_json_filepath, 'rb'), purpose = 'batch')
    batch_input_file_id = batch_input_file.id

    batch_log = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
        metadata={
          "description": description
        }
    )
    print(f'Batch log created: {batch_log.id}')

    return batch_input_file_id, batch_log


'''prepare batch input & run batch & retrieve batch output'''

def prepare_batch_input(df, input_prompt_template, system_prompt='You are a helpful assistant',
                        template_properties={}, model='gpt-4o',
                        temperature=1.0, max_tokens=1024, input_filepath = None):
    assert input_filepath is not None, 'input_filepath is required'

    if os.path.exists(input_filepath):
        os.remove(input_filepath)

    for i in tqdm(range(len(df))):
        if len(template_properties) > 0:
            properties = {k: df[v].iloc[i] if v in df.columns else v for k, v in template_properties.items()}

        input_prompt = input_prompt_template.format(**properties)

        query = batch_query_template(
            input_prompt=input_prompt,
            system_prompt=system_prompt,
            model=model,
            custom_id=f'idx_{df.index[i]}',
            temperature=temperature,
            max_tokens=max_tokens
        )

        cache_batch_query(input_filepath, query)

def run_model (df, input_prompt_template, system_prompt, template_properties,
              model, temperature, max_tokens, batch_size, input_filepath, batch_log_filepath,
              cache_filepath, mode, batch_rate_limit = None):
    '''prepare batch input'''

    if model == 'gpt-4o':
        batch_rate_limit = 5
        
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
                stream_generate_response(input_filepath=input_filepath, cache_filepath=cache_filepath)
                print(f'Results are generated and stored at {cache_filepath}')
            
        else:   
            minibatch_stream_generate_response(input_filepath=input_filepath, batch_log_filepath=batch_log_filepath, batch_size = batch_size, batch_rate_limit = batch_rate_limit)
            print(f'Results are generated and check {batch_log_filepath}')

def retrieve_outputs (batch_log_filepath, cache_filepath):
    if os.path.exists(cache_filepath):
        print(f'Results are retrieved from {cache_filepath}')
        output_df = pd.read_pickle(cache_filepath)
    else:
        with open(batch_log_filepath) as f:
            batch_logs = json.load(f)

        output_dict = {}
        for idx, batch_log_id in batch_logs.items():
            status = check_batch_status(batch_log_id)
            if status == 'completed':
                output_file_id = retrieve_batch_output_file_id(batch_log_id)
                output_dict[idx] = output_file_id
            else:
                print(f'Batch {batch_log_id} at {idx} failed')

        output_df = minibatch_retrieve_response(output_dict_filepath=None, output_dict=output_dict)
        output_df.to_pickle(cache_filepath)
        print(f'Results are retrieved and stored at {cache_filepath}')

    return output_df

'''utils for batch'''

def retrieve_batch_output_file_id(batch_log_id: str):
    batch_log = client.batches.retrieve(batch_log_id)
    return batch_log.output_file_id

def check_batch_status(batch_log_id: str):
    batch_log = client.batches.retrieve(batch_log_id)
    print(f'Batch log status: {batch_log.status}')

    return batch_log.status

def check_batch_error(batch_log_id: str):
    batch_log = client.batches.retrieve(batch_log_id)
    if batch_log.status == 'failed':
        print(f'Batch {batch_log_id} failed with error: {batch_log.errors}')
        return batch_log.errors
    else:
        return None
    
def cancel_batch(batch_log_id: str):
    client.batches.cancel(batch_log_id)
    return f'Batch {batch_log_id} is cancelled'

def batch_query_template(input_prompt: str, system_prompt: str = 'You are a helpful assistant', model: str = 'gpt-4o', custom_id: str = None,
                temperature: float = 0.0, max_tokens: int = 1024, top_p: float = 1.0, frequency_penalty: float = 0.0,
                presence_penalty: float = 0.0, stop: list[str] = None):
    query_template = {"custom_id": custom_id,
                  "method": "POST",
                  "url": "/v1/chat/completions",
                   "body": {"model": model,
                            "temperature": temperature,
                            "messages": [{"role": "system", "content": system_prompt},
                                         {"role": "user", "content": input_prompt}
                                        ],
                            "max_tokens": max_tokens,
                            "top_p": top_p,
                            "frequency_penalty": frequency_penalty,
                            "presence_penalty": presence_penalty,
                            "stop": stop}
                 }
    return query_template

def cache_batch_query(filepath: str, query: dict):
    with open(filepath, 'a') as f:
        f.write(json.dumps(query) + '\n')

def load_batch_query(filepath: str):
    with open(filepath, 'r') as f:
        for line in f:
            yield json.loads(line)

