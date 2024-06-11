#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/6/11 11:16
"""
import sys
sys.path.append('.')
sys.path.append('/home/test/codes/GoMate')
import datasets
from langchain_core.documents import Document
from tqdm import tqdm

from gomate.modules.retrieval.embedding import TextEmbedding
from gomate.modules.retrieval.faiss_retriever import FaissRetriever, FaissRetrieverConfig

if __name__ == '__main__':
    dataset_path = '/home/test/codes/GoMate/data/docs/wikipedia-nq-corpus'
    wikipedia_dataset = datasets.load_from_disk(dataset_path)
    print(len(wikipedia_dataset['train']))
    embedding_model_path = "/home/test/pretrained_models/mpnet-base"
    embedding_model = TextEmbedding(embedding_model_path,batch_size=256)
    retriever_config = FaissRetrieverConfig(
        embedding_model=embedding_model,
        top_k=5,
        embedding_model_string="mpnet-base",
        vectorstore_path="wikipedia_index",
        rebuild_index=True
    )
    faiss_retriever = FaissRetriever(config=retriever_config)
    documents = []
    for sample in tqdm(wikipedia_dataset['train']):
        documents.append(
            Document(
                page_content=sample['text'],
                meta_data={
                    'docid': sample['docid'],
                    'title': sample['title']
                }
            )
        )
    faiss_retriever.build_from_documents(documents)
    contexts = faiss_retriever.retrieve("Aaron is a prophet")
    print(contexts)
