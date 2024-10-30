import requests
import json
import pandas as pd
import os


def get_es_data():
    os.makedirs('data/', exist_ok=True)
    keywords = [
        "美国",
        "中美贸易",
        "俄乌冲突",
        "中东",

    ]

    for word in keywords:
        url = f'http://10.208.61.117:9200/document_share_data_30_news/_search?q={word}&size=6000&sort=publish_time:desc'
        response = requests.get(url)
        with open(f'data/{word}_data.json', 'w', encoding='utf-8') as f:
            json.dump(response.json(), f, ensure_ascii=False, indent=4)

        with open(f'data/{word}_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        sources = [hit['_source'] for hit in data['hits']['hits']]

        source_df = pd.DataFrame(sources)

        source_df['id'] = source_df.index
        source_df['id'] = 'source_' + source_df['id'].astype(str)


        source_df[['id', 'title', 'content']].to_excel(f'data/{word}_data.xlsx')
