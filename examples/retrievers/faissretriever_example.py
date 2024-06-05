#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/6/5 22:36
"""
import json

from gomate.modules.retrieval.embedding import SBertEmbeddingModel
from gomate.modules.retrieval.faiss_retriever import FaissRetriever, FaissRetrieverConfig

if __name__ == '__main__':
    from transformers import AutoTokenizer

    embedding_model_path = "/home/test/pretrained_models/bge-large-zh-v1.5"
    embedding_model = SBertEmbeddingModel(embedding_model_path)
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
    retriever_config = FaissRetrieverConfig(
        max_tokens=100,
        max_context_tokens=3500,
        use_top_k=True,
        embedding_model=embedding_model,
        top_k=5,
        tokenizer=tokenizer,
        embedding_model_string="bge-large-zh-v1.5",
        index_path="faiss_index.bin",
        rebuild_index=True
    )

    faiss_retriever = FaissRetriever(config=retriever_config)

    documents = []
    with open('/home/test/codes/GoMate/data/zh_refine.json', 'r', encoding="utf-8") as f:
        for line in f.readlines():
            data = json.loads(line)
            documents.extend(data['positive'])
            documents.extend(data['negative'])
    print(len(documents))
    faiss_retriever.build_from_texts(documents[:200])

    contexts = faiss_retriever.retrieve("2022年冬奥会开幕式总导演是谁")
    print(contexts)
