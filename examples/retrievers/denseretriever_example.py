#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: denseretriever_example.py
@time: 2024/06/06
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import os

import pandas as pd

from trustrag.modules.document.utils import PROJECT_BASE
from trustrag.modules.retrieval.dense_retriever import DenseRetriever, DenseRetrieverConfig

if __name__ == '__main__':
    print(PROJECT_BASE)
    retriever_config = DenseRetrieverConfig(
        model_name_or_path="H:/pretrained_models/mteb/bge-large-zh-v1.5",
        dim=1024,
        index_path=os.path.join(PROJECT_BASE, 'output/weibo_dense'),
    )
    config_info = retriever_config.log_config()
    print(config_info)
    retriever = DenseRetriever(config=retriever_config)
    # data = pd.read_json('/data/users/searchgpt/yq/GoMate/data/docs/zh_refine.json', lines=True)[:5]
    # print(data)
    # print(data.columns)
    #
    # corpus = []
    # for documents in tqdm(data['positive'], total=len(data)):
    #     for document in documents:
    #         # retriever.add_text(document)
    #         corpus.append(document)
    # for documents in tqdm(data['negative'], total=len(data)):
    #     for document in documents:
    #         #     retriever.add_text(document)
    #         corpus.append(document)
    # print("len(corpus)", len(corpus))
    # retriever.build_from_texts(corpus)
    # result = retriever.retrieve("RCEP具体包括哪些国家")
    # print(result)
    # retriever.save_index()
    data = pd.read_json(
        os.path.join(PROJECT_BASE, 'data/docs/weibo/2024031123-2feb535f-a4b4-432d-98c6-eb1c60ddcd1f.data'),
        lines=True)
    data['text'] = data['title'] + ' ' + data['content']
    # retriever.build_from_texts(data['text'].tolist())
    # retriever.save_index()

    retriever.load_index()
    result = retriever.retrieve("演员刘畅")
    print(result)
