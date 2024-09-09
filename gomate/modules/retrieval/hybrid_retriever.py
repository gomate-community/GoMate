#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@ Hybrid Retriever
@ Author: 
@ Contact: 
@ License: Apache Licence
@ Time: 2024/8/27 14:08
"""

from typing import List, Dict

from gomate.modules.retrieval.bm25s_retriever import BM25Retriever, BM25RetrieverConfig
from gomate.modules.retrieval.dense_retriever import DenseRetriever, DenseRetrieverConfig
from gomate.modules.retrieval.base import BaseRetriever, BaseConfig


class HybridRetrieverConfig(BaseConfig):
    """
        Configuration class for setting up a hybrid retriever.

        Attributes:
            bm25_config (BM25RetrieverConfig): Configuration for the BM25 retriever.
            dense_config (DenseRetrieverConfig): Configuration for the Dense retriever.
            bm25_weight (float): Weight for the BM25 scores in the final hybrid score.
            dense_weight (float): Weight for the Dense retriever scores in the final hybrid score.
    """

    def __init__(self, bm25_config, dense_config, bm25_weight=0.5, dense_weight=0.5):
        self.bm25_config = bm25_config
        self.dense_config = dense_config
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight


class HybridRetriever(BaseRetriever):
    """
        Implements a hybrid retriever combining BM25 and dense retrieval.

        Methods:
            __init__(config): Initializes the hybrid retriever with given configurations.
            build_from_texts(texts): Processes and indexes a list of texts using both retrievers.
            retrieve(query, top_k): Retrieves the top_k documents using the hybrid method.
    """

    def __init__(self, config):
        self.bm25_retriever = BM25Retriever(config.bm25_config)
        self.dense_retriever = DenseRetriever(config.dense_config)
        self.bm25_weight = config.bm25_weight
        self.dense_weight = config.dense_weight

    def save_index(self):
        self.bm25_retriever.save_index()
        self.dense_retriever.save_index()

    def load_index(self):
        self.bm25_retriever.load_index()
        self.dense_retriever.load_index()

    def build_from_texts(self, texts: List[str] = None):
        """Build indexes for both BM25 and dense retrievers."""
        if texts is not None:
            self.bm25_retriever.build_from_texts(texts)
            self.dense_retriever.build_from_texts(texts)

    def retrieve(self, query: str = None, top_k: int = 5) -> List[Dict]:
        """Retrieve top_k documents using hybrid BM25 + dense method."""
        # Retrieve documents from both BM25 and dense retrievers
        bm25_results = self.bm25_retriever.retrieve(query, top_k)
        dense_results = self.dense_retriever.retrieve(query, top_k)

        # Combine results from both methods
        hybrid_scores = {}
        for result in bm25_results:
            hybrid_scores[result['text']] = self.bm25_weight * result['score']

        for result in dense_results:
            if result['text'] in hybrid_scores:
                hybrid_scores[result['text']] += self.dense_weight * result['score']
            else:
                hybrid_scores[result['text']] = self.dense_weight * result['score']

        # Sort documents by their combined scores
        sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

        # Format the results to include text and score
        return [{'text': text, 'score': score} for text, score in sorted_results[:top_k]]
