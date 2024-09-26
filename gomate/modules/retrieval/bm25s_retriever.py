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
import numpy as np
from tqdm import tqdm

from gomate.modules.document.rag_tokenizer import tokenize
from gomate.modules.retrieval.base import BaseConfig, BaseRetriever

jieba.setLogLevel(logging.INFO)


def process_sentence(sent='', tokenizer_func='rag'):
    if tokenizer_func == 'jieba':
        return ' '.join([w for w in jieba.cut(sent)])
    else:
        return ' '.join(tokenize(sent))


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
            b=0.75,
            tokenizer_func='rag'
    ):
        self.method = method
        self.index_path = index_path
        self.k1 = k1
        self.b = b
        self.tokenizer_func = tokenizer_func

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
    def __init__(self, config):
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
        self.load_mode = True
        self.retriever = bm25s.BM25.load(self.index_path, load_corpus=True)
        self.corpus = self.retriever.corpus

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
        self.load_mode = False

    def retrieve(self, query: str = None, top_k: int = 5):
        query = process_sentence(query)
        query_tokens = bm25s.tokenize(
            query,
            token_pattern=r'(?u)\b\w+\b',
            stopwords=None,
            stemmer=self.stemmer_fn
        )

        indexes, scores = self.retriever.retrieve(query_tokens, k=top_k)

        # Flatten and convert to list
        indexes = indexes.flatten().tolist() if isinstance(indexes, np.ndarray) else indexes[0]
        scores = scores.flatten().tolist() if isinstance(scores, np.ndarray) else scores[0]

        result = []
        for i, (index, score) in enumerate(zip(indexes, scores)):
            if isinstance(index, dict):
                doc_id = index['id']
            else:
                doc_id = index

            if self.load_mode:
                text = self.corpus[doc_id]['text']
            else:
                text = self.corpus[doc_id]

            result.append({'text': text, 'score': score})

        return result
