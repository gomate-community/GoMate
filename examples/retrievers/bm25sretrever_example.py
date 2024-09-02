#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: 
@contact:
@license: Apache Licence
@time: 2024/08/27 14:16
"""
import os

from gomate.modules.document.common_parser import CommonParser
from gomate.modules.retrieval.bm25s_retriever import BM25Retriever

if __name__ == '__main__':


    corpus = []

    new_files = [
        r'/data/users/searchgpt/yq/GoMate_dev/data/docs/伊朗.txt',
        r'/data/users/searchgpt/yq/GoMate_dev/data/docs/伊朗总统罹难事件.txt',
        r'/data/users/searchgpt/yq/GoMate_dev/data/docs/伊朗总统莱希及多位高级官员遇难的直升机事故.txt',
        r'/data/users/searchgpt/yq/GoMate_dev/data/docs/伊朗问题.txt',
        r'/data/users/searchgpt/yq/GoMate_dev/data/docs/新冠肺炎疫情.pdf',
    ]
    parser = CommonParser()
    for filename in new_files:
        chunks = parser.parse(filename)
        corpus.extend(chunks)
    bm25_retriever = BM25Retriever(method="lucene",
                                   index_path="indexs/description_bm25.index",
                                   rebuild_index=True,
                                   corpus=corpus)
    query = "新冠疫情"
    search_docs = bm25_retriever.retrieve(query)
    print(search_docs)