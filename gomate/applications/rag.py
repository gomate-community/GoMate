#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: RagApplication.py
@time: 2024/05/20
@contact: yanqiangmiffy@gamil.com
"""
from gomate.modules.document.reader import ReadFiles
from gomate.modules.generator.llm import GLMChat
from gomate.modules.retrieval.embedding import BgeEmbedding
from gomate.modules.store import VectorStore

class RagApplication():
    def __init__(self, config):
        self.config=config
        self.vector_store = VectorStore([])
        self.llm = GLMChat(config.llm_model_name)
        self.reader = ReadFiles(config.docs_path)
        self.embedding_model = BgeEmbedding(config.embedding_model_name)
    def init_vector_store(self):
        docs=self.reader.get_content(max_token_len=600, cover_content=150)
        self.vector_store.document=docs
        self.vector_store.get_vector(EmbeddingModel=self.embedding_model)
        self.vector_store.persist(path='storage')  # 将向量和文档内容保存到storage目录下，下次再用就可以直接加载本地的数据库
        self.vector_store.load_vector(path='storage')  # 加
    def load_vector_store(self):
        self.vector_store.load_vector(path=self.config.vector_store_path)  # 加载本地的数据库

    def add_document(self, file_path):
        docs = self.reader.get_content_by_file(file=file_path, max_token_len=512, cover_content=60)
        self.vector_store.add_documents(self.config.vector_store_path, docs, self.embedding_model)

    def chat(self, question: str = '', topk: int = 5):
        contents = self.vector_store.query(question, EmbeddingModel=self.embedding_model, k=topk)
        content = '\n'.join(contents[:5])
        print(contents)
        response, history = self.llm.chat(question, [], content)
        return response, history,contents
