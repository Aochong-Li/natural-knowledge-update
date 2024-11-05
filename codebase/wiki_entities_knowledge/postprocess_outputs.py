import pandas as pd
from tqdm import tqdm
import re

def parse_fact (text: str, keyword = 'Fact'):
    fact_idx = text.index(keyword)

    return text[fact_idx : ]

def parse_facts_from_step1v (df: pd.DataFrame, keyword = 'Fact'):
    output_df = df.copy()
    output_df['step1v_output'] = output_df['response'].str.split('\n')

    output_df['step1v_output'] = output_df['step1v_output'].apply(lambda x: [fact for fact in x if (keyword in fact and 'Not Supported'.lower() not in fact.lower())])
    output_df['step1v_output'] = output_df['step1v_output'].str.replace('Fact:', '')

    return output_df['step1v_output']

def filter_facts_from_step1c (df: pd.DataFrame):
    output_df = df.copy()
    output_df['step1c_output'] = output_df['response'].apply(lambda x: x \
                                                             if ('True' in x.split('Output:')[1]) and ('False' not in x.split('Output:')[1]) \
                                                                else None)

    return output_df[output_df['step1c_output'].notnull()][['step1c_output']]

def parse_reason_from_step2 (df: pd.DataFrame, col: str = 'response'):
    output_df = df.copy()
    
    output_df = output_df[(~output_df[col].str.contains('fact is not changeable'))
                          & (output_df[col].str.contains('Transition:')
                          & (output_df[col].str.contains('New Fact:')))]
    
    output_df['step2_new_fact'] = output_df[col].str.split('New Fact:').str[1]
    output_df['step2_reasoning'] = output_df[col].str.split('New Fact:').str[0]
    
    output_df = output_df[output_df['step2_reasoning'].str.contains('Transition:')]
    output_df['step2_reasoning'] = output_df['step2_reasoning'].str.replace('Transition:', '')

    return output_df.drop(columns=[col])

def parse_article_from_step3 (df: pd.DataFrame, col: str = 'response'):
    output_df = df.copy()
    output_df = output_df[output_df[col].str.contains('[Article]')]
    output_df['step3_article'] = output_df[col].str.split('[Article]:').str[1]

    return output_df.drop(columns=[col])

'''Deprecated'''
def parse_facts_from_step1 (df: pd.DataFrame, exp_num_facts: int = 5):
    output_df = df.copy()
    output_df['facts'] = None

    for i in tqdm(range(len(df))):
        facts = df['response'].iloc[i]
        facts = facts.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
        facts = [fact for fact in facts.split('\n') if ('Fact' in fact) and ('"' in fact)]
        pattern = r'"(.*)"'
        facts = sum([re.findall(pattern, fact) for fact in facts], [])
        output_df['facts'].iloc[i] = facts

    output_df['num_facts'] = output_df['facts'].str.len()
    output_df = output_df[output_df['num_facts'] == exp_num_facts]
    output_df.drop(columns=['num_facts'], inplace = True)

    return output_df

def get_fact_context (wikitext: str, fact: str):
    # parse a wikitext into paragraphs
    # only keep paragraphs that are longer than 5 words
    wikiparagraphs = [paragraph for paragraph in wikitext.split('\n') if len(paragraph.split(' ')) > 5]

    fact = fact[:-1] if fact.endswith('.') else fact
    first_half_fact, second_half_fact = fact[:len(fact)//2], fact[len(fact)//2:]
    paragraphs_with_fact = [paragraph for paragraph in wikiparagraphs if first_half_fact in paragraph or second_half_fact in paragraph]

    if len(paragraphs_with_fact) == 0:
        return wikitext
    
    # paragraph_indicies = [wikiparagraphs.index(paragraph) for paragraph in paragraphs_with_fact]
    # wiki_context = []
    # for i in paragraph_indicies:
    #     fact_context = wikiparagraphs[max(0, i - 1): min(len(wikiparagraphs) - 1, i + 2)]
    #     fact_context = '\n'.join(fact_context) if len(fact_context) > 1 else fact_context[0]
    #     wiki_context.append(fact_context)

    result = '\n'.join(paragraphs_with_fact) if len(paragraphs_with_fact) > 1 else paragraphs_with_fact[0]
    return result
        