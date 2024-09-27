#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author:
@contact:
@license: Apache Licence
@time: 2024/8/27 14:07
"""
import logging
from multiprocessing import Pool
from typing import List, Union, Callable, Dict, Any

import bm25s
import jieba
import numpy as np
from tqdm import tqdm

from gomate.modules.document.rag_tokenizer import tokenize
from gomate.modules.retrieval.base import BaseConfig, BaseRetriever

jieba.setLogLevel(logging.INFO)


def process_sentence(sent: str = '', tokenizer_func: str = 'rag') -> str:
    """
    tokenize sentence
    """
    # print(tokenizer_func)
    if tokenizer_func == 'jieba':
        return ' '.join([w for w in jieba.cut(sent)])
    elif tokenizer_func == 'rag':
        return ' '.join(tokenize(sent))
    else:
        raise ValueError(f"Unsupported tokenizer function: {tokenizer_func}")


class BM25RetrieverConfig(BaseConfig):
    """
    Configuration class for BM25 Retriever.

    Attributes:
        method (str): The retrieval method, e.g., 'robertson', 'lucene', 'atire', 'bm25l', 'bm25+'
        index_path (str): Path to save or load the BM25 index.
        rebuild_index (bool): Flag to rebuild the index if True.
        k1 (float): BM25 hyperparameter controlling term saturation.
        b (float): BM25 hyperparameter controlling length normalization.
        tokenizer_func (str): The tokenizer function to use ('jieba' or 'rag').
        num_processes (int): Number of processes to use for multiprocessing.
    """

    def __init__(
            self,
            method: str = "lucene",
            index_path: str = "indexs/description_bm25.index",
            k1: float = 1.5,
            b: float = 0.75,
            tokenizer_func: str = 'jieba',
            num_processes: int = 16,
            rebuild_index: bool = False
    ):
        self.method = method
        self.index_path = index_path
        self.k1 = k1
        self.b = b
        self.tokenizer_func = tokenizer_func
        self.num_processes = num_processes
        self.rebuild_index = rebuild_index

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
        if self.tokenizer_func not in ['jieba', 'rag']:
            raise ValueError("tokenizer_func must be either 'jieba' or 'rag'.")
        if not isinstance(self.num_processes, int) or self.num_processes <= 0:
            raise ValueError("num_processes must be a positive integer.")
        if not isinstance(self.rebuild_index, bool):
            raise ValueError("rebuild_index must be a boolean.")
        print("BM25 configuration is valid.")


class BM25Retriever(BaseRetriever):
    def __init__(self, config: BM25RetrieverConfig):
        super().__init__()
        self.config = config
        self.method = config.method
        self.index_path = config.index_path
        self.stemmer_fn: Callable[[List[str]], List[str]] = lambda lst: [word for word in lst]
        self.retriever: Union[bm25s.BM25, None] = None
        self.corpus: Union[List[str], None] = None
        self.load_mode: bool = False


    def process_corpus(self, corpus: List[str]) -> List[str]:
        # print(self.config.tokenizer_func)
        print("tokenizeing...")
        with Pool(processes=self.config.num_processes) as pool:
            processed_corpus = list(
                pool.starmap(process_sentence, [(sent, self.config.tokenizer_func) for sent in corpus])
            )
        print("tokenized done!")
        return processed_corpus


    def save_index(self):
        if self.retriever is not None:
            self.retriever.save(self.index_path, corpus=self.corpus)
        else:
            raise ValueError("Retriever is not initialized. Build or load an index first.")

    def load_index(self):
        self.load_mode = True
        self.retriever = bm25s.BM25.load(self.index_path, load_corpus=True)
        self.corpus = self.retriever.corpus

    def build_from_texts(self, corpus: List[str]):
        print("build_from_texts...")
        self.corpus = corpus
        processed_corpus = self.process_corpus(corpus)
        corpus_tokens = bm25s.tokenize(
            processed_corpus,
            token_pattern=r'(?u)\b\w+\b',
            stopwords=None,
            stemmer=self.stemmer_fn
        )
        self.retriever = bm25s.BM25(method=self.method, k1=self.config.k1, b=self.config.b)
        self.retriever.index(corpus_tokens)
        self.load_mode = False

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.retriever is None:
            raise ValueError("Retriever is not initialized. Build or load an index first.")

        processed_query = process_sentence(query, self.config.tokenizer_func)
        query_tokens = bm25s.tokenize(
            processed_query,
            token_pattern=r'(?u)\b\w+\b',
            stopwords=None,
            stemmer=self.stemmer_fn
        )

        indexes, scores = self.retriever.retrieve(query_tokens, k=top_k)

        indexes = indexes.flatten().tolist() if isinstance(indexes, np.ndarray) else indexes[0]
        scores = scores.flatten().tolist() if isinstance(scores, np.ndarray) else scores[0]

        result = []
        for index, score in zip(indexes, scores):
            doc_id = index['id'] if isinstance(index, dict) else index
            text = self.corpus[doc_id]['text'] if self.load_mode else self.corpus[doc_id]
            result.append({'text': text, 'score': score})

        return result
