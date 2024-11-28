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

import pandas as pd

from trustrag.modules.citation.match_citation import MatchCitation
from trustrag.modules.document.utils import PROJECT_BASE
from trustrag.modules.generator.llm import GLM4Chat
from trustrag.modules.reranker.bge_reranker import BgeReranker, BgeRerankerConfig
from trustrag.modules.retrieval.dense_retriever import DenseRetriever, DenseRetrieverConfig


class ApplicationConfig():
    def __init__(self):
        self.retriever_config = None
        self.rerank_config = None


class WeiboRagApplication():
    def __init__(self, config):
        self.config = config
        self.retriever = DenseRetriever(self.config.retriever_config)
        self.reranker = BgeReranker(self.config.rerank_config)
        self.llm = GLM4Chat(self.config.llm_model_path)
        self.mc = MatchCitation()

    def init_vector_store(self):
        """

        """
        print("init_vector_store ... ")
        data = pd.read_json(
            os.path.join(PROJECT_BASE, 'data/docs/weibo/2024031123-2feb535f-a4b4-432d-98c6-eb1c60ddcd1f.data'),
            lines=True)
        data['text'] = data['title'] + ' ' + data['content']

        self.retriever.build_from_texts(data['text'].tolist())
        print("init_vector_store done! ")
        self.retriever.save_index(self.config.retriever_config.index_dir)

    def load_vector_store(self):
        self.retriever.load_index(self.config.retriever_config.index_path)

    def add_document(self, file_path):
        data = pd.read_json(file_path, lines=True)
        data['text'] = data['title'] + ' ' + data['content']
        for chunk in data['text'].tolist():
            self.retriever.add_text(chunk)
        print("add_document done!")

    def chat(self, question: str = '', top_k: int = 20):
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


# if __name__ == '__main__':
#     # 修改成自己的配置！！！
#     app_config = ApplicationConfig()
#     app_config.llm_model_path = "H:/pretrained_models/llm/glm-4-9b-chat"
#
#     retriever_config = DenseRetrieverConfig(
#         model_name_or_path="H:/pretrained_models/mteb/bge-large-zh-v1.5",
#         dim=1024,
#         index_path=os.path.join(PROJECT_BASE, 'output/weibo_dense')
#     )
#     rerank_config = BgeRerankerConfig(
#         model_name_or_path="H:/pretrained_models/mteb/bge-reranker-large"
#     )
#
#     app_config.retriever_config = retriever_config
#     app_config.rerank_config = rerank_config
#     application = WeiboRagApplication(app_config)
#     # application.init_vector_store()
#     application.load_vector_store()
#     result, history, contents = application.chat("刘畅演员最近有什么活动")
#     print(result, history, contents)