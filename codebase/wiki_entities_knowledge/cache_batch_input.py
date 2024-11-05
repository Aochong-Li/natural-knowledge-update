import prompts
import openaiAPI
from tqdm import tqdm
import os


def prepare_batch_input(df, input_prompt_template, system_prompt=prompts.SYSTEM_PROMPT,
                        template_properties={}, model='gpt-4o',
                        temperature=1.0, max_tokens=1024, input_filepath = None):
    assert input_filepath is not None, 'input_filepath is required'

    if os.path.exists(input_filepath):
        os.remove(input_filepath)

    for i in tqdm(range(len(df))):
        if len(template_properties) > 0:
            properties = {k: df[v].iloc[i] if v in df.columns else v for k, v in template_properties.items()}

        input_prompt = input_prompt_template.format(**properties)

        query = openaiAPI.batch_query_template(
            input_prompt=input_prompt,
            system_prompt=system_prompt,
            model=model,
            custom_id=f'idx_{df.index[i]}',
            temperature=temperature,
            max_tokens=max_tokens
        )

        openaiAPI.cache_batch_query(input_filepath, query)