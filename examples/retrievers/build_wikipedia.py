#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/6/11 11:16
"""
import sys
import gc
sys.path.append('.')
sys.path.append('/home/test/codes/GoMate')
import datasets
from langchain_core.documents import Document
from tqdm import tqdm
from langchain_community.vectorstores.faiss import FAISS

from gomate.modules.retrieval.embedding import TextEmbedding
from gomate.modules.retrieval.faiss_retriever import FaissRetriever, FaissRetrieverConfig


def process_batch(batch, batch_index):
    embedding_model_path = "/home/test/pretrained_models/mpnet-base"
    embedding_model = TextEmbedding(embedding_model_path, batch_size=256)
    retriever_config = FaissRetrieverConfig(
        embedding_model=embedding_model,
        top_k=5,
        embedding_model_string="mpnet-base",
        vectorstore_path=f"wikipedia_index_{batch_index}",
        rebuild_index=True
    )
    faiss_retriever = FaissRetriever(config=retriever_config)
    documents = []
    print(batch[0])
    for sample in tqdm(batch):
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
    del faiss_retriever
    gc.collect()

if __name__ == '__main__':

    build=False
    if build:

        dataset_path = '/home/test/codes/GoMate/data/docs/wikipedia-nq-corpus'
        wikipedia_dataset = datasets.load_from_disk(dataset_path)
        print(len(wikipedia_dataset['train']))

        batch_size = 2000000
        num_batches = (len(wikipedia_dataset['train']) + batch_size - 1) // batch_size

        for batch_index in range(num_batches):
            start_idx = batch_index * batch_size
            end_idx = min(start_idx + batch_size, len(wikipedia_dataset['train']))
            batch = wikipedia_dataset['train'].select(range(start_idx,end_idx))
            process_batch(batch, batch_index)
            del batch

    merge=False
    if merge:
        embedding_model_path = "/home/test/pretrained_models/mpnet-base"
        embedding_model = TextEmbedding(embedding_model_path, batch_size=256)
        db = FAISS.load_local('/home/test/codes/GoMate/wikipedia_index_0', embeddings=embedding_model, index_name='index',allow_dangerous_deserialization=True)
        for i in tqdm(range(2,11)):
            db_new = FAISS.load_local(f'/home/test/codes/GoMate/wikipedia_index_{i}', embeddings=embedding_model,
                                  index_name='index', allow_dangerous_deserialization=True)

            db.merge_from(db_new)
        db.save_local('/home/test/codes/GoMate/wikipedia_index')


    embedding_model_path = "/home/test/pretrained_models/mpnet-base"
    embedding_model = TextEmbedding(embedding_model_path, batch_size=256)
    db = FAISS.load_local('/home/test/codes/GoMate/wikipedia_index', embeddings=embedding_model, index_name='index',allow_dangerous_deserialization=True)
    docs = db.similarity_search_with_score('Iphone')
    print(docs)