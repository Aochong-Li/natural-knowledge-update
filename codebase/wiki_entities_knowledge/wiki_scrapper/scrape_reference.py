import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

import pandas as pd
import os
import sys
import json
from tqdm import tqdm
import re

from codebase.wiki_entities_knowledge.wiki_scrapper.constants import unwanted_extensions

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) ' 
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/129.0.0.0 Safari/537.36'
}

def grab_reference_articles(df: pd.DataFrame, url_col: str = 'url', title_col: str = 'title', cache_filepath: str = None):
    output_df = df[[url_col, title_col]].drop_duplicates()
    output_df['wiki_reference'] = None

    for i in tqdm(range(len(output_df[url_col]))):
        url = output_df[url_col].iloc[i]

        reference_content_dict = {}
        reference_urls = get_reference_urls(url)

        print(f'retreived {len(reference_urls)} reference urls for {url}')
        
        for reference_url in tqdm(reference_urls):
            content = get_content_from_reference_url(reference_url)
            if len(content) == 0:
                continue
            reference_content_dict[reference_url] = content
        
        output_df['wiki_reference'].iloc[i] = reference_content_dict

        output_df.to_pickle(cache_filepath)

# TODO: temporary function

def get_reference_retrieved_date (wiki_url):
    response = requests.get(wiki_url, headers=headers, timeout=10)
    response.raise_for_status()

    links_dates = {}

    soup = BeautifulSoup(response.content, 'html.parser')
    ref_cols = soup.select('div.reflist')
    
    if len(ref_cols) == 0:
        return links_dates
    
    for ref_col in ref_cols:
        references = ref_col.find('ol', class_='references')
        li_elements = references.find_all('li', id = True)

        for li in li_elements:
            '''get date'''
            text = li.get_text()
            if 'Retrieved' not in text:
                continue
            else:
                text = text.split('Retrieved')[1]
                pattern = r'(.*)\b(\d{4})\b'
                match = re.search(pattern, text)
                if not match:
                    continue
                else:
                    date = match.group(1) + ' ' + match.group(2)


            reference_text = li.find('span', class_='reference-text')
            if reference_text:
                a_tags = reference_text.find_all('a', href = True)
                for a_tag in a_tags:
                    href = a_tag['href']
                    # TODO: check the filtering on links makes sense
                    # TODO: add date here
                    if (href.startswith('http://') or href.startswith('https://')) \
                        and (not any(href.lower().endswith(ext) for ext in unwanted_extensions))\
                        and (not 'web.archive.org' in href):

                        links_dates[href] = date

    return links_dates
    
    
def get_reference_urls(wiki_url):
    response = requests.get(wiki_url, headers=headers, timeout=10)
    response.raise_for_status()

    links = []

    soup = BeautifulSoup(response.content, 'html.parser')
    ref_cols = soup.select('div.reflist')
    
    if len(ref_cols) == 0:
        return links
    
    for ref_col in ref_cols:
        references = ref_col.find('ol', class_='references')
        li_elements = references.find_all('li', id = True)

        for li in li_elements:
            reference_text = li.find('span', class_='reference-text')
            if reference_text:
                a_tags = reference_text.find_all('a', href = True)
                for a_tag in a_tags:
                    href = a_tag['href']
                    # TODO: check the filtering on links makes sense
                    # TODO: add date here
                    if (href.startswith('http://') or href.startswith('https://')) \
                        and (not any(href.lower().endswith(ext) for ext in unwanted_extensions))\
                        and (not 'web.archive.org' in href):

                        links.append(href)

    return list(set(links))

def get_content_from_reference_url(reference_url):
    from newspaper import Article

    try:
        article = Article(reference_url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        # print(f"Error parsing {reference_url}: {e}")
        return ""