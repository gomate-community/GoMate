#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/6/11 11:16
"""
from tqdm import tqdm
from datasets import load_dataset
import datasets
from langchain_core.documents import Document
from gomate.modules.retrieval.embedding import SBertEmbedding
from gomate.modules.retrieval.faiss_retriever import FaissRetriever,FaissRetrieverConfig
if __name__ == '__main__':
    dataset_path = '/home/test/codes/GoMate/data/docs/wikipedia-nq-corpus'
    wikipedia_dataset = datasets.load_from_disk(dataset_path)
    print(len(wikipedia_dataset['train']))
    embedding_model_path = "/home/test/pretrained_models/mpnet-base"
    embedding_model = SBertEmbedding(embedding_model_path)
    retriever_config = FaissRetrieverConfig(
        embedding_model=embedding_model,
        top_k=5,
        embedding_model_string="bge-large-zh-v1.5",
        vectorstore_path="zh_refine_index",
        rebuild_index=True
    )
    faiss_retriever = FaissRetriever(config=retriever_config)
    documents = []
    with open('/home/test/codes/GoMate/data/zh_refine.json', 'r', encoding="utf-8") as f:
        for sample in tqdm(wikipedia_dataset['train']):
            documents.append(
                Document(
                    page_content=sample['text'],
                    meta_data={
                        'docid':sample['docid'],
                        'title':sample['title']
                    }
                )
            )
    faiss_retriever.build_from_documents(documents)
    contexts = faiss_retriever.retrieve("首趟RCEP班列的起点")
    print(contexts)
