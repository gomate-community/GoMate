#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: 
@contact: 
@license: Apache Licence
@time: 2024/08/27 14:16
"""

from gomate.modules.document.common_parser import CommonParser
from gomate.modules.retrieval.bm25s_retriever import BM25RetrieverConfig
from gomate.modules.retrieval.dense_retriever import DenseRetrieverConfig
from gomate.modules.retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverConfig

if __name__ == '__main__':
    # BM25 and Dense Retriever configurations
    bm25_config = BM25RetrieverConfig(
        method='lucene',
        index_path='indexs/description_bm25.index',
        k1=1.6,
        b=0.7
    )
    bm25_config.validate()
    print(bm25_config.log_config())

    dense_config = DenseRetrieverConfig(
        model_name_or_path="/data/users/searchgpt/pretrained_models/bge-large-zh-v1.5",
        dim=1024,
        index_path='/data/users/searchgpt/yq/GoMate/examples/retrievers/dense_cache'
    )
    config_info = dense_config.log_config()
    print(config_info)

    # Hybrid Retriever configuration
    hybrid_config = HybridRetrieverConfig(bm25_config=bm25_config, dense_config=dense_config, bm25_weight=0.5,
                                          dense_weight=0.5)
    hybrid_retriever = HybridRetriever(config=hybrid_config)

    # Corpus
    corpus = []

    # Files to be parsed
    new_files = [
        r'/data/users/searchgpt/yq/GoMate_dev/data/docs/伊朗.txt',
        r'/data/users/searchgpt/yq/GoMate_dev/data/docs/伊朗总统罹难事件.txt',
        r'/data/users/searchgpt/yq/GoMate_dev/data/docs/伊朗总统莱希及多位高级官员遇难的直升机事故.txt',
        r'/data/users/searchgpt/yq/GoMate_dev/data/docs/伊朗问题.txt',
        r'/data/users/searchgpt/yq/GoMate_dev/data/docs/新冠肺炎疫情.pdf',
    ]

    # Parsing documents
    parser = CommonParser()
    for filename in new_files:
        chunks = parser.parse(filename)
        corpus.extend(chunks)

    # Build hybrid retriever from texts
    # hybrid_retriever.build_from_texts(corpus)
    # hybrid_retriever.save_index()
    hybrid_retriever.load_index()
    # Query
    query = "新冠肺炎疫情"
    results = hybrid_retriever.retrieve(query, top_k=3)

    # Output results
    for result in results:
        print(f"Text: {result['text']}, Score: {result['score']}")
