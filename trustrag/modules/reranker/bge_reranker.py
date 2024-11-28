#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: bge_reranker.py
@time: 2024/06/05
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import List, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from trustrag.modules.reranker.base import BaseReranker


class BgeRerankerConfig:
    """
    Configuration class for setting up a BERT-based reranker.

    Attributes:
        model_name_or_path (str): Path or model identifier for the pretrained model from Hugging Face's model hub.
        device (str): Device to load the model onto ('cuda' or 'cpu').
    """

    def __init__(self, model_name_or_path='bert-base-uncased'):
        self.model_name_or_path = model_name_or_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def log_config(self):
        # Log the current configuration settings
        return f"""
        BgeRerankerConfig:
            Model Name or Path: {self.model_name_or_path}
            Device: {self.device}
        """


class BgeReranker(BaseReranker):
    """
    A reranker that utilizes a BERT-based model for sequence classification
    to rerank a list of documents based on their relevance to a given query.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(config.model_name_or_path) \
            .half().to(config.device).eval()
        self.device = config.device
        print('Successful load rerank model')

    def rerank(self, query: str, documents: List[str], k: int = 5, is_sorted: bool = True) -> list[dict[str, Any]]:
        # Process input documents for uniqueness and formatting
        # documents = list(set(documents))
        pairs = [[query, d] for d in documents]

        # Tokenize and predict relevance scores
        with torch.no_grad():
            inputs = self.rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt',
                                           max_length=512).to(self.device)
            scores = self.rerank_model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist()

        # Pair documents with their scores, sort by scores in descending order
        if is_sorted:
            ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            # Return the top k documents
            top_docs = [{'text': doc, 'score': score} for doc, score in ranked_docs]
        else:
            top_docs = [{'text': doc, 'score': score} for doc, score in zip(documents, scores)]
        return top_docs
