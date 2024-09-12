#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: 
@contact:
@license: Apache Licence
@time: 2024/08/27 14:16
"""
import os

import pandas as pd

from gomate.modules.document.utils import PROJECT_BASE
from gomate.modules.retrieval.bm25s_retriever import BM25RetrieverConfig, BM25Retriever

if __name__ == '__main__':
    # corpus = []

    # new_files = [
    #     f'{PROJECT_BASE}/data/docs/伊朗.txt',
    #     f'{PROJECT_BASE}/data/docs/伊朗总统罹难事件.txt',
    #     f'{PROJECT_BASE}/data/docs/伊朗总统莱希及多位高级官员遇难的直升机事故.txt',
    #     f'{PROJECT_BASE}/data/docs/伊朗问题.txt',
    #     f'{PROJECT_BASE}/data/docs/汽车操作手册.pdf',
    #     # r'H:\2024-Xfyun-RAG\data\corpus.txt'
    # ]
    # parser = CommonParser()
    # for filename in new_files:
    #     chunks = parser.parse(filename)
    #     corpus.extend(chunks)
    data = pd.read_json(
        os.path.join(PROJECT_BASE, 'data/docs/weibo/2024031123-2feb535f-a4b4-432d-98c6-eb1c60ddcd1f.data'),
        lines=True)
    data['text'] = data['title'] + ' ' + data['content']

    bm25_config = BM25RetrieverConfig(
        method='lucene',
        index_path=f'{PROJECT_BASE}/output/weibo_dense',
        k1=1.6,
        b=0.7
    )
    bm25_config.validate()
    print(bm25_config.log_config())

    bm25_retriever = BM25Retriever(bm25_config)
    corpus = data['text'].tolist()
    bm25_retriever.build_from_texts(corpus)
    # bm25_retriever.load_index()
    query = "刘畅"
    search_docs = bm25_retriever.retrieve(query)
    print(search_docs)
