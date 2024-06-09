#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/6/8 1:25
"""
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document

documents = [
    Document(
        meta_data={'text': 'PC'},
        page_content='个人电脑',
    ),
    Document(
        meta_data={'text': 'doctor'},
        page_content='医生办公室',
    )
]
embedding_path = r'H:\pretrained_models\bert\english\paraphrase-multilingual-mpnet-base-v2'
embedding_model = HuggingFaceEmbeddings(model_name=embedding_path)
db = FAISS.from_documents(documents, embedding=embedding_model)

db.save_local('../.cache/faiss.index')

db = FAISS.load_local('../.cache/faiss.index', embeddings=embedding_model, index_name='index',allow_dangerous_deserialization=True)
docs = db.similarity_search_with_score('台式机电脑')
print(docs)

