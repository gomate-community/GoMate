#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: vectorbase.py
@time: 2024/05/23
@contact: yanqiangmiffy@gamil.com
"""
import json
import os
from typing import List

import numpy as np
from tqdm import tqdm

from gomate.modules.retrieval.embedding import BaseEmbeddings, BgeEmbedding


class VectorStore:
    def __init__(self, document: List[str] = ['']) -> None:
        self.document = document

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:

        self.vectors = []
        for doc in tqdm(self.document, desc="Calculating embeddings"):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors

    def persist(self, path: str = 'storage'):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/doecment.json", 'w', encoding='utf-8') as f:
            json.dump(self.document, f, indent=2, ensure_ascii=False)
        if self.vectors:
            with open(f"{path}/vectors.json", 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f, indent=2, ensure_ascii=False)

    def load_vector(self, path: str = 'storage'):
        with open(f"{path}/vectors.json", 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)
        with open(f"{path}/doecment.json", 'r', encoding='utf-8') as f:
            self.document = json.load(f)

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        return BaseEmbeddings.cosine_similarity(vector1, vector2)

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
        query_vector = EmbeddingModel.get_embedding(query)
        result = np.array([self.get_similarity(query_vector, vector)
                           for vector in self.vectors])
        return np.array(self.document)[result.argsort()[-k:][::-1]].tolist()

    def add_documents(
            self,
            path: str = 'storage',
            documents: List[str] = [''],
            EmbeddingModel: BaseEmbeddings = BgeEmbedding
    ) -> List[List[float]]:
        # load existed vector
        self.load_vector(path)
        for doc in documents:
            self.document.append(doc)
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        print("len(self.document),len(self.vectors):", len(self.document), len(self.vectors))
        self.persist(path)
