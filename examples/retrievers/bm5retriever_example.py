#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/6/1 15:48
"""
from gomate.modules.retrieval.bm25_retriever import BM25RetrieverConfig, BM25Retriever, tokenizer

if __name__ == '__main__':
    bm25_retriever_config = BM25RetrieverConfig(
        tokenizer=tokenizer,
        k1=1.5,
        b=0.75,
        epsilon=0.25,
        delta=0.25,
        algorithm='Okapi'
    )
    bm25_retriever = BM25Retriever(bm25_retriever_config)
    corpus = [

    ]
    new_files = [
        r'H:\Projects\GoMate\data\伊朗.txt',
        r'H:\Projects\GoMate\data\伊朗总统罹难事件.txt',
        r'H:\Projects\GoMate\data\伊朗总统莱希及多位高级官员遇难的直升机事故.txt',
        r'H:\Projects\GoMate\data\伊朗问题.txt',
    ]
    for filename in new_files:
        with open(filename, 'r', encoding="utf-8") as file:
            corpus.append(file.read())
    bm25_retriever.fit_bm25(corpus)
    query = "伊朗总统莱希"
    search_docs = bm25_retriever.retrieve(query)
    print(search_docs)
