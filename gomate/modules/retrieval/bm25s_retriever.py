#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author:
@contact:
@license: Apache Licence
@time: 2024/8/27 14:07
"""
import logging
from multiprocessing import Pool, cpu_count

import bm25s
import jieba
from tqdm import tqdm

from gomate.modules.retrieval.base import BaseConfig, BaseRetriever

jieba.setLogLevel(logging.INFO)


def process_sentence(sent):
    return ' '.join([w for w in jieba.cut(sent)])


class BM25RetrieverConfig(BaseConfig):
    """
    Configuration class for BM25 Retriever.

    Attributes:
        method (str): The retrieval method, e.g., 'robertson', 'lucene', 'atire', 'bm25l', 'bm25+'
        index_path (str): Path to save or load the BM25 index.
        rebuild_index (bool): Flag to rebuild the index if True.
        k1 (float): BM25 hyperparameter controlling term saturation.
        b (float): BM25 hyperparameter controlling length normalization.
    """

    def __init__(
            self,
            method="lucene",
            index_path="indexs/description_bm25.index",
            k1=1.5,
            b=0.75
    ):
        self.method = method
        self.index_path = index_path
        self.k1 = k1
        self.b = b

    def validate(self):
        """Validate BM25 configuration parameters."""
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("Method must be a non-empty string.")
        if not isinstance(self.index_path, str) or not self.index_path:
            raise ValueError("Index path must be a non-empty string.")
        if not isinstance(self.k1, (int, float)) or self.k1 <= 0:
            raise ValueError("k1 must be a positive number.")
        if not isinstance(self.b, (int, float)) or not (0 <= self.b <= 1):
            raise ValueError("b must be a number between 0 and 1.")
        print("BM25 configuration is valid.")


class BM25Retriever(BaseRetriever):
    def __init__(self,config):
        super().__init__()
        self.method = config.method
        self.index_path = config.index_path
        self.stemmer_fn = lambda lst: [word for word in lst]
        self.retriever = None

    def process_corpus(self, corpus, num_processes=16):
        if num_processes is None:
            num_processes = cpu_count()

        with Pool(processes=num_processes) as pool:
            processed_corpus = list(tqdm(
                pool.imap(process_sentence, corpus),
                total=len(corpus),
                desc="Processing sentences"
            ))

        return processed_corpus

    def save_index(self):
        self.retriever.save(self.index_path, corpus=self.corpus)

    def load_index(self):
        self.retriever = bm25s.BM25.load(self.index_path, load_corpus=True)

    def build_from_texts(self, corpus):
        print("build_from_texts...")
        self.corpus = corpus
        corpus = self.process_corpus(corpus)
        corpus_tokens = bm25s.tokenize(
            corpus,
            token_pattern=r'(?u)\b\w+\b',
            stopwords=None,
            stemmer=self.stemmer_fn
        )
        # Create the BM25 model and index the corpus
        self.retriever = bm25s.BM25(method=self.method)
        self.retriever.index(corpus_tokens)
        # You can save the corpus along with the model
        # self.retriever = bm25s.BM25.load(self.index_path, load_corpus=True)

    def retrieve(self, query: str = None, top_k: int = 5):
        # Query the corpus
        query = process_sentence(query)
        query_tokens = bm25s.tokenize(
            query,
            token_pattern=r'(?u)\b\w+\b',
            stopwords=None,
            stemmer=self.stemmer_fn
        )
        # print(query_tokens)
        # query_tokens = bm25s.tokenize(query, stopwords=None,stemmer=None)
        # print(query_tokens)
        # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
        # results, scores = retriever.retrieve(query_tokens, corpus=corpus, k=5)
        indexes, scores = self.retriever.retrieve(query_tokens, k=top_k)
        # print(indexes, type(indexes))
        indexes = indexes.tolist()

        print(indexes[0], type(indexes[0]))
        # print(scores)
        indices = []
        similarities = []
        # 判断 indexes 是二维结构还是 list[dict] 结构,load_index模式
        if isinstance(indexes[0], dict):
            # 如果 indexes 是 list[dict] 结构
            for i in range(len(indexes)):
                doc, score = indexes[i], scores[i]
                indices.append(doc['id'])
                similarities.append(score)
        else:
            #
            # 如果 indexes 是嵌套列表
            for i in range(len(indexes[0])):  # 使用 len(indexes[0]) 来获取内层列表的长度
                doc_id, score = indexes[0][i], scores[0][i]
                indices.append(doc_id)
                similarities.append(score)

        # 返回一个包含文档文本和得分的列表
        return [{'text': self.corpus[indices[i]], 'score': similarities[i]} for i in range(len(indices))]
