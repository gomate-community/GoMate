#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: RagApplication.py
@time: 2024/05/20
@contact: yanqiangmiffy@gamil.com
"""
import os

from gomate.modules.citation.match_citation import MatchCitation
from gomate.modules.document.common_parser import CommonParser
from gomate.modules.generator.llm import GLMChat
from gomate.modules.reranker.bge_reranker import BgeReranker
from gomate.modules.retrieval.dense_retriever import DenseRetriever


class ApplicationConfig():
    def __init__(self):
        self.retriever_config = None
        self.rerank_config = None


class RagApplication():
    def __init__(self, config):
        self.config = config
        self.parser = CommonParser()
        self.retriever = DenseRetriever(self.config.retriever_config)
        self.reranker = BgeReranker(self.config.rerank_config)
        self.llm = GLMChat(self.config.llm_model_path)
        self.mc = MatchCitation()

    def init_vector_store(self):
        """

        """
        print("init_vector_store ... ")
        chunks = []
        for filename in os.listdir(self.config.docs_path):
            file_path = os.path.join(self.config.docs_path, filename)
            try:
                chunks.extend(self.parser.parse(file_path))
            except:
                pass
        self.retriever.build_from_texts(chunks)
        print("init_vector_store done! ")
        self.retriever.save_index(self.config.retriever_config.index_dir)

    def load_vector_store(self):
        self.retriever.load_index(self.config.retriever_config.index_dir)

    def add_document(self, file_path):
        chunks = self.parser.parse(file_path)
        for chunk in chunks:
            self.retriever.add_text(chunk)
        print("add_document done!")

    def chat(self, question: str = '', top_k: int = 5):
        contents = self.retriever.retrieve(query=question, top_k=top_k)
        contents = self.reranker.rerank(query=question, documents=[content['text'] for content in contents])
        content = '\n'.join([content['text'] for content in contents])
        print(contents)
        response, history = self.llm.chat(question, [], content)
        result = self.mc.ground_response(
            response=response,
            evidences=[content['text'] for content in contents],
            selected_idx=[idx for idx in range(len(contents))],
            markdown=True
        )
        return result, history, contents
