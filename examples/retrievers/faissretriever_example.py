#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/6/5 22:36
"""
import json

from trustrag.modules.retrieval.embedding import SBertEmbeddingModel
from trustrag.modules.retrieval.faiss_retriever import FaissRetriever, FaissRetrieverConfig

if __name__ == '__main__':
    embedding_model_path = "/data/users/searchgpt/pretrained_models/bge-large-zh-v1.5"
    embedding_model = SBertEmbeddingModel(embedding_model_path)
    retriever_config = FaissRetrieverConfig(
        embedding_model=embedding_model,
        embedding_model_string="bge-large-zh-v1.5",
        index_path="/data/users/searchgpt/yq/GoMate/examples/retrievers/faiss_index.bin",
        rebuild_index=True
    )
    faiss_retriever = FaissRetriever(config=retriever_config)
    documents = []
    with open('/data/users/searchgpt/yq/GoMate/data/docs/zh_refine.json', 'r', encoding="utf-8") as f:
        for line in f.readlines():
            data = json.loads(line)
            documents.extend(data['positive'])
            documents.extend(data['negative'])
    faiss_retriever.build_from_texts(documents[:200])
    search_contexts = faiss_retriever.retrieve("2021年香港GDP增长了多少")
    print(search_contexts)
