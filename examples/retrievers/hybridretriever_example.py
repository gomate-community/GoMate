#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: 
@contact: 
@license: Apache Licence
@time: 2024/08/27 14:16
"""

import os
import pickle

import pandas as pd
from tqdm import tqdm

from trustrag.modules.document.chunk import TextChunker
from trustrag.modules.document.common_parser import CommonParser
from trustrag.modules.document.utils import PROJECT_BASE
from trustrag.modules.retrieval.bm25s_retriever import BM25RetrieverConfig
from trustrag.modules.retrieval.dense_retriever import DenseRetrieverConfig
from trustrag.modules.retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverConfig
from trustrag.modules.generator.llm import PROMPT_TEMPLATE,SYSTEM_PROMPT

def process_chunk(text):
    """
    文本预处理
    """
    text = text.replace(
        '本文档为2024CCFBDCI比赛用语料的一部分。部分文档使用大语言模型改写生成，内容可能与现实情况不符，可能不具备现实意义，仅允许在本次比赛中使用。',
        '')
    text = text.replace(
        '本文档为2024 CCF BDCI 比赛用语料的一部分。部分文档使用大语言模型改写生成，内容可能与现实情况不符，可能不具备现实意义，仅允许在本次比赛中使用。',
        '')
    return text


def generate_chunks():
    cp = CommonParser()
    tc = TextChunker()
    chunks = []
    for pdf_file in os.listdir(f'{PROJECT_BASE}/data/competitions/df/texts'):
        print(pdf_file)
        if pdf_file.endswith('.txt'):
            paragraphs = cp.parse(f'{PROJECT_BASE}/data/competitions/df/texts/{pdf_file}')
            chunk = tc.chunk_sentences(paragraphs, chunk_size=512)
            chunks.append(chunk)

    with open(f'{PROJECT_BASE}/output/df_chunks.pkl', 'wb') as f:
        print(len(chunks))
        pickle.dump(chunks, f)
    pd.DataFrame(data={'chunks':chunks}).to_csv(f'{PROJECT_BASE}/output/chunks.csv',index=False)

if __name__ == '__main__':

    test_path = "/data/users/searchgpt/yq/GoMate_dev/data/competitions/df/A_question.csv"
    embedding_model_path = "/data/users/searchgpt/pretrained_models/bge-large-zh-v1.5"
    llm_model_path = "/data/users/searchgpt/pretrained_models/glm-4-9b-chat"
    # ====================文件解析+切片=========================
    # generate_chunks()

    with open(f'{PROJECT_BASE}/output/df_chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)
    corpus = []
    for chunk_list in chunks:
        for chunk in chunk_list:
            chunk = process_chunk(chunk)
            if chunk.strip():
                corpus.append(chunk.strip())
    pd.DataFrame(data={'corpus':corpus}).to_csv(f'{PROJECT_BASE}/output/corpus.csv',index=False)

    # ====================检索器配置=========================
    # BM25 and Dense Retriever configurations
    bm25_config = BM25RetrieverConfig(
        method='lucene',
        index_path='indexs/df_description_bm25.index',
        k1=1.6,
        b=0.7,

    )
    bm25_config.validate()
    print(bm25_config.log_config())
    dense_config = DenseRetrieverConfig(
        model_name_or_path=embedding_model_path,
        dim=1024,
        index_path='indexs/df_dense_cache'
    )
    config_info = dense_config.log_config()
    print(config_info)
    # Hybrid Retriever configuration
    # 由于分数框架不在同一维度，建议可以合并
    hybrid_config = HybridRetrieverConfig(
        bm25_config=bm25_config,
        dense_config=dense_config,
        bm25_weight=0.7,  # bm25检索结果权重
        dense_weight=0.3  # dense检索结果权重
    )
    hybrid_retriever = HybridRetriever(config=hybrid_config)
    # 构建索引
    # hybrid_retriever.build_from_texts(corpus)
    # 保存索引
    # hybrid_retriever.save_index()
    # 加载索引
    hybrid_retriever.load_index()

    # ====================检索测试=========================
    query = "支付宝"
    results = hybrid_retriever.retrieve(query, top_k=10)
    print(len(results))
    # Output results
    for result in results:
        print(f"Text: {result['text']}, Score: {result['score']}")

    # ====================检索问答=========================
    test = pd.read_csv(test_path)
    answers = []
    contexts = []

    for query in tqdm(test['question'], total=len(test)):
        tmp=[]
        search_docs = hybrid_retriever.retrieve(query, top_k=5)
        context = ""
        for i, doc in enumerate(search_docs):
            context += f"信息[{i}]{doc['text']}\n\n"
        print(context)
        contexts.append(context)
        for result in search_docs:
            tmp.append(result['text'])
        answers.append(tmp)
    test['answers']=answers
    test['contexts']=contexts
    test.to_excel(f'{PROJECT_BASE}/output/answers.xlsx', index=False)