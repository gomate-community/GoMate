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
from typing import List, Dict
import jieba
import bm25s  # Import bm25s library
from gomate.modules.retrieval.base import BaseRetriever
jieba.setLogLevel(logging.INFO)

def tokenizer(text: str):
    return [word for word in jieba.cut(text)]

class BM25RetrieverConfig:
    def __init__(self, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25, delta=0.5, algorithm='Okapi'):
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.delta = delta
        self.algorithm = algorithm

    def log_config(self):
        config_summary = """
            BM25RetrieverConfig:
                Tokenizer: {tokenizer},
                K1: {k1},
                B: {b},
                Epsilon: {epsilon},
                Delta: {delta},
                Algorithm: {algorithm},
            """.format(
            tokenizer=self.tokenizer,
            k1=self.k1,
            b=self.b,
            epsilon=self.epsilon,
            delta=self.delta,
            algorithm=self.algorithm,
        )
        return config_summary


class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        self.tokenizer = config.tokenizer
        self.k1 = config.k1
        self.b = config.b
        self.epsilon = config.epsilon
        self.delta = config.delta
        self.algorithm = config.algorithm

    def build_from_texts(self, corpus):
        self.corpus = corpus
        self.corpus_tokens = bm25s.tokenize(corpus, tokenizer=self.tokenizer)  # Tokenize the corpus
        
        if self.algorithm == 'Okapi':
            self.bm25 = bm25s.BM25(corpus=self.corpus_tokens, method='BM25Okapi', k1=self.k1, b=self.b)
        elif self.algorithm == 'BM25L':
            self.bm25 = bm25s.BM25(corpus=self.corpus_tokens, method='BM25L', k1=self.k1, b=self.b, delta=self.delta)
        elif self.algorithm == 'BM25Plus':
            self.bm25 = bm25s.BM25(corpus=self.corpus_tokens, method='BM25Plus', k1=self.k1, b=self.b, delta=self.delta)
        elif self.algorithm == 'ATIRE':
            self.bm25 = bm25s.BM25(corpus=self.corpus_tokens, method='atire', k1=self.k1, b=self.b)
        elif self.algorithm == 'Lucene':
            self.bm25 = bm25s.BM25(corpus=self.corpus_tokens, method='lucene', k1=self.k1, b=self.b)
        else:
            raise ValueError('Algorithm not supported')

        self.bm25.index(self.corpus_tokens)  # Index the corpus

    def retrieve(self, query: str = '', top_k: int = 3) -> List[Dict]:
        tokenized_query = bm25s.tokenize([query], tokenizer=self.tokenizer)[0]
        indices, scores = self.bm25.retrieve(tokenized_query, k=top_k)
        search_docs = [{'text': self.corpus[i], 'score': scores[0, i]} for i in indices[0]]
        return search_docs
