import pageviewapi
import pandas as pd
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import pickle
import wikipediaapi
AGENT = 'knowledge_udpate (13641361467@gmail.com)'
WIKI = wikipediaapi.Wikipedia(AGENT, 'en')

def get_page_summary (page_title: str):
    page = WIKI.page(page_title)
    
    if not page.exists():
        return ""
    else:
        return page.summary
    
def add_pagesummary_to_dataset (dataset_path = "", cache_file_path = ""):
    assert cache_file_path != "", "cache_file_path must be provided"
    result = {}

    '''get dataset'''
    ds = load_from_disk(dataset_path=dataset_path)

    for item in tqdm(ds):
        page_summary = get_page_summary(item["title"])
        if page_summary != "":
            result[item["title"]] = page_summary
        else:
            continue

        if len(result) % 1000 == 0:
            with open(cache_file_path, "wb") as f:
                pickle.dump(result, f)

def get_monthly_avg_pageviews(page_title: str, start_date: str = '20221201', end_date: str = '20231130') -> int:

    try:
        views = pageviewapi.per_article(
                                        'en.wikipedia',      # The language and project (e.g., 'en.wikipedia')
                                        page_title,               # The title of the Wikipedia page
                                        start_date,          # Start date in 'YYYYMMDD' format
                                        end_date,            # End date in 'YYYYMMDD' format
                                        access='all-access', # 'all-access', 'desktop', 'mobile-app', or 'mobile-web'
                                        agent='all-agents',  # 'all-agents', 'user', 'spider', or 'bot'
                                        granularity='monthly'  # 'daily' or 'monthly'
                                    )
        if len(views['items']) != 12:
            return 0
        else:
            return int(pd.DataFrame(views['items'])['views'].mean())
    except Exception as e:
        return 0

def add_pageviews_to_dataset (dataset_path = "", cache_file_path = ""):
    assert cache_file_path != "", "cache_file_path must be provided"

    result = {}

    '''get dataset'''
    ds = load_from_disk(dataset_path=dataset_path)

    for item in tqdm(ds):
        page_view = get_monthly_avg_pageviews(item["title"])
        if page_view > 0:
            result[item["title"]] = page_view
        else:
            continue

        if len(result) % 1000 == 0:
            with open(cache_file_path, "wb") as f:
                pickle.dump(result, f)


if __name__ == "__main__":
    dataset_path = "/share/goyal/lio/dataset/huggingface/wikipedia_20231101.en_sample_n=200000_seed=42"

    cache_file_path = "/share/goyal/lio/knowledge_update/wikipedia/wiki_pagesummary/20221201_20231130_page_summary_samplen=200000.pickle"
    add_pagesummary_to_dataset(dataset_path=dataset_path, cache_file_path=cache_file_path)

    # cache_file_path = "/share/goyal/lio/knowledge_update/wikipedia/wiki_pageview/monthly_avg_20221201_20231130_page_view_samplen=200000.pickle"
    # add_pageviews_to_dataset(dataset_path=dataset_path, cache_file_path=cache_file_path)