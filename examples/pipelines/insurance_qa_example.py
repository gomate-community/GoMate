#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: insurance_qa_example.py
@time: 2024/06/07
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import json
import pandas as pd
from transformers import AutoTokenizer
from gomate.modules.retrieval.embedding import SBertEmbeddingModel
from gomate.modules.retrieval.faiss_retriever import FaissRetriever
from gomate.modules.retrieval.faiss_retriever import FaissRetrieverConfig

## step1 build faiss index
embedding_model_path = "/home/test/pretrained_models/bge-large-zh-v1.5"
embedding_model = SBertEmbeddingModel(embedding_model_path)
tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
retriever_config = FaissRetrieverConfig(
    max_tokens=1024,
    max_context_tokens=8192,
    use_top_k=True,
    embedding_model=embedding_model,
    top_k=5,
    tokenizer=tokenizer,
    embedding_model_string="bge-large-zh-v1.5",
    index_path="faiss_index.bin",
    rebuild_index=True
)

faiss_retriever = FaissRetriever(config=retriever_config)

# step2 加载文件
data=pd.read_json('../../data/competitions/round1_training_data/test.json')
documents = []
for idx,row in data.iterrows():
    documents.append(row['条款'])
print(documents)
faiss_retriever.build_from_texts(documents)

contexts = faiss_retriever.retrieve("确诊严重运动神经元病需要哪些条件？")
print(contexts)
