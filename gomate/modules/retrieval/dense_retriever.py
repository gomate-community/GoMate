#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/6/5 23:07
"""
import gc
import os
import shutil
from typing import List

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from gomate.modules.retrieval.base import BaseConfig,BaseRetriever


# class DenseRetrieverConfig(BaseConfig):
#     """
#         Configuration class for setting up a dense retriever.
#
#         Attributes:
#             model_name (str): Name of the transformer model to be used.
#             dim (int): Dimension of the embeddings.
#             top_k (int): Number of top results to retrieve.
#             index_path (str, optional): Path to save or load the FAISS index.
#             rebuild_index (bool): Flag to rebuild the index if True.
#     """
#
#     def __init__(
#             self,
#             model_name_or_path='sentence-transformers/all-mpnet-base-v2',
#             dim=768,
#             index_path=None,
#             rebuild_index=True
#     ):
#         self.model_name = model_name_or_path
#         self.dim = dim
#         self.index_path = index_path
#         self.rebuild_index = rebuild_index
#
#     def log_config(self):
#         # Create a formatted string that summarizes the configuration
#         config_summary = f"""
#         DenseRetrieverConfig:
#         Model Name: {self.model_name}
#         Dimension: {self.dim}
#         Index Path: {self.index_path}
#         Rebuild Index: {'Yes' if self.rebuild_index else 'No'},
#         """
#         return config_summary

class DenseRetrieverConfig(BaseConfig):
    """
    Configuration class for Dense Retriever.

    Attributes:
        model_name (str): Name of the transformer model to be used.
        dim (int): Dimension of the embeddings.
        index_path (str): Path to save or load the FAISS index.
        rebuild_index (bool): Flag to rebuild the index if True.
    """

    def __init__(self, model_name_or_path='sentence-transformers/all-mpnet-base-v2', dim=768, index_path=None):
        self.model_name = model_name_or_path
        self.dim = dim
        self.index_path = index_path

    def validate(self):
        """Validate Dense configuration parameters."""
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("Model name must be a non-empty string.")
        if not isinstance(self.dim, int) or self.dim <= 0:
            raise ValueError("Dimension must be a positive integer.")
        if self.index_path and not isinstance(self.index_path, str):
            raise ValueError("Index directory path must be a string.")
        print("Dense configuration is valid.")

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
        self.dim = config.dim
        self.embeddings = []
        self.documents = []
        self.num_documents = 0
        self.index_path = config.index_path

        # self.rebuild_index = config.rebuild_index
        # if self.rebuild_index and self.index_path and os.path.exists(self.index_path):
        #     shutil.rmtree(self.index_path)
        #     # os.remove(self.index_path)
        #     # Rebuild the index
        #     self.index = faiss.IndexFlatIP(self.dim)
        # elif not self.rebuild_index and self.index_path and os.path.exists(self.index_path):
        #     self.load_index(self.index_path)
        # else:
        #     # Initialize a new index
        #     self.index = faiss.IndexFlatIP(self.dim)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt',max_length=512)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.detach().numpy()

    def load_index(self, index_path: str = None):
        """Load the FAISS index from the specified path."""
        if index_path is None:
            index_path = self.index_path
        data = np.load(os.path.join(index_path, 'document.vecstore.npz'), allow_pickle=True)
        self.documents, self.embeddings = data['documents'].tolist(), data['embeddings'].tolist()
        self.index = faiss.read_index(os.path.join(index_path, 'fassis.index'))
        print("Index loaded successfully from", index_path)
        del data
        gc.collect()

    def save_index(self, index_path: str = None):
        """Save the FAISS index to the specified path."""
        if self.index and self.embeddings and self.documents:
            if index_path is None:
                index_path = self.index_path
            if not os.path.exists(index_path):
                os.makedirs(index_path,exist_ok=True)
                print(f"Index saving to：{index_path}")
            np.savez(
                os.path.join(index_path, 'document.vecstore'),
                embeddings=self.embeddings,
                documents=self.documents
            )
            faiss.write_index(self.index, os.path.join(index_path, 'fassis.index'))
            print("Index saved successfully to", index_path)

    def add_text(self, text):
        # Ensure single document is processed as a list
        embedding = self.get_embedding([text])
        self.index.add(embedding)
        self.documents.append(text)
        self.embeddings.append(embedding)
        self.num_documents += 1

    def build_from_texts(self, corpus: List[str] = None):
        if corpus is None:
            return
        # # Batch processing of texts to improve efficiency
        # embeddings = self.get_embedding(texts)
        # self.index.add(embeddings)
        # for text in texts:
        #     self.doc_map[self.num_documents] = text
        #     self.num_documents += 1
        # # Save the index after building
        for text in tqdm(corpus,desc="build_from_texts.."):
            self.add_text(text)

    def retrieve(self, query: str = None, top_k: int = 5):
        D, I = self.index.search(self.get_embedding([query]), top_k)
        return [{'text':self.documents[idx],'score':score} for idx, score in zip(I[0], D[0])]
