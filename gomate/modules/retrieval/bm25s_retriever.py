#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author:
@contact:
@license: Apache Licence
@time: 2024/8/27 14:07
"""
import logging
import os
import shutil
from multiprocessing import Pool, cpu_count

import bm25s
import jieba
from tqdm import tqdm

jieba.setLogLevel(logging.INFO)


def process_sentence(sent):
    return ' '.join([w for w in jieba.cut(sent)])


class BM25Retriever():
    def __init__(
            self,
            method="lucene",
            index_path="indexs/description_bm25.index",
            rebuild_index=True,
            corpus=None
    ):
        super().__init__()
        self.method = method
        self.index_path = index_path
        self.rebuild_index = rebuild_index
        self.stemmer_fn = lambda lst: [word for word in lst]
        self.corpus=corpus
        # Load the index from the specified path if it is not None
        if not self.rebuild_index:
            if self.index_path and os.path.exists(self.index_path):
                self.load_index()
            else:
                self.build_from_texts(corpus)
        else:
            print("rebuild_index")
            if os.path.exists(self.index_path):
                shutil.rmtree(self.index_path)
            self.build_from_texts(corpus)

    def load_index(self):
        self.retriever = bm25s.BM25.load(self.index_path, load_corpus=True)

    def process_corpus(self, corpus, num_processes=32):
        if num_processes is None:
            num_processes = cpu_count()

        with Pool(processes=num_processes) as pool:
            processed_corpus = list(tqdm(
                pool.imap(process_sentence, corpus),
                total=len(corpus),
                desc="Processing sentences"
            ))

        return processed_corpus

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
        self.retriever.save(self.index_path, corpus=corpus)
        self.retriever = bm25s.BM25.load(self.index_path, load_corpus=True)

    def retrieve(self, query=None, top_k=5):
        # Query the corpus
        query=process_sentence(query)
        query_tokens = bm25s.tokenize(
            query,
            token_pattern=r'(?u)\b\w+\b',
            stopwords=None,
            stemmer=self.stemmer_fn
        )
        print(query_tokens)
        # query_tokens = bm25s.tokenize(query, stopwords=None,stemmer=None)
        # print(query_tokens)
        # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
        # results, scores = retriever.retrieve(query_tokens, corpus=corpus, k=5)
        indexs, scores = self.retriever.retrieve(query_tokens, k=top_k)
        # print(indexs,type(indexs))
        print(scores)
        indices = []
        similarities = []
        for i in range(indexs.shape[1]):
            doc, score = indexs[0, i], scores[0, i]
            indices.append(doc['id'])
            similarities.append(score)
        return [{'text': self.corpus[indices[i]], 'score': similarities[i]} for i in range(indexs.shape[1])]
