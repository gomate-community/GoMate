#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/6/5 23:07
"""
import os

import faiss
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from gomate.modules.retrieval.base import BaseRetriever


class DenseRetrieverConfig:
    """
        Configuration class for setting up a dense retriever.

        Attributes:
            model_name (str): Name of the transformer model to be used.
            dim (int): Dimension of the embeddings.
            top_k (int): Number of top results to retrieve.
            index_path (str, optional): Path to save or load the FAISS index.
            rebuild_index (bool): Flag to rebuild the index if True.
    """
    def __init__(
        self,
        model_name='sentence-transformers/all-mpnet-base-v2',
        dim=768,
        top_k=3,
        index_path=None,
        rebuild_index=True
    ):
        self.model_name = model_name
        self.dim = dim
        self.top_k=top_k
        self.index_path = index_path
        self.rebuild_index = rebuild_index

    def log_config(self):
        # Create a formatted string that summarizes the configuration
        config_summary = f"""
        DenseRetrieverConfig:
            Model Name: {self.model_name}
            Dimension: {self.dim}
            TOP_K:{self.top_k}
            Index Path: {self.index_path}
            Rebuild Index: {'Yes' if self.rebuild_index else 'No'},
            
        """
        return config_summary


class DenseRetriever(BaseRetriever):
    """
        Implements a dense retriever for efficiently searching documents.

        Methods:
            __init__(config): Initializes the retriever with given configuration.
            mean_pooling(model_output, attention_mask): Performs mean pooling on model outputs.
            get_embedding(sentences): Generates embeddings for provided sentences.
            load_index(index_path): Loads the FAISS index from a file.
            save_index(): Saves the current FAISS index to a file.
            add_doc(document_text): Adds a document to the index.
            build_from_texts(texts): Processes and indexes a list of texts.
            retrieve(query): Retrieves the top_k documents relevant to the query.
    """
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name)
        self.index = faiss.IndexFlatIP(config.dim)
        self.dim=config.dim
        self.top_k=config.top_k
        self.doc_map = dict()
        self.ctr = 0

        self.index_path = config.index_path
        self.rebuild_index = config.rebuild_index
        if self.rebuild_index and self.index_path and os.path.exists(self.index_path):
            os.remove(self.index_path)
            self.index = faiss.IndexFlatIP(self.dim)  # Rebuild the index
        elif not self.rebuild_index and self.index_path and os.path.exists(self.index_path):
            self.load_index(self.index_path)
        else:
            # Initialize a new index
            self.index = faiss.IndexFlatIP(self.dim)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.detach().numpy()

    def load_index(self, index_path):
        """Load the FAISS index from the specified path."""
        self.index = faiss.read_index(index_path)
        print("Index loaded successfully from", index_path)

    def save_index(self):
        """Save the FAISS index to the specified path."""
        if self.index and self.index_path:
            faiss.write_index(self.index, self.index_path)
            print("Index saved successfully to", self.index_path)
    def add_doc(self, document_text):
        self.index.add(self.get_embedding([document_text]))  # Ensure single document is processed as a list
        self.doc_map[self.ctr] = document_text
        self.ctr += 1

    def build_from_texts(self, texts: List[str]=None):
        if texts is None:
            return
        # Batch processing of texts to improve efficiency
        embeddings = self.get_embedding(texts)
        print(embeddings.shape)
        self.index.add(embeddings)
        for text in texts:
            self.doc_map[self.ctr] = text
            self.ctr += 1
        # Save the index after building
        self.save_index()
    def retrieve(self, query):
        print(self.top_k)
        D, I = self.index.search(self.get_embedding([query]), self.top_k)
        return [{self.doc_map[idx]: score} for idx, score in zip(I[0], D[0]) if idx in self.doc_map]
