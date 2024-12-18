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
from typing import List

import faiss
import numpy as np
from FlagEmbedding import FlagModel
from tqdm import tqdm

from trustrag.modules.retrieval.base import BaseConfig, BaseRetriever

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DenseRetrieverConfig(BaseConfig):
    """
    Configuration class for Dense Retriever.

    Attributes:
        model_name (str): Name of the transformer model to be used.
        dim (int): Dimension of the embeddings.
        index_path (str): Path to save or load the FAISS index.
        rebuild_index (bool): Flag to rebuild the index if True.
    """

    def __init__(
            self,
            model_name_or_path='sentence-transformers/all-mpnet-base-v2',
            dim=768,
            index_path=None,
            batch_size=32
    ):
        self.model_name = model_name_or_path
        self.dim = dim
        self.index_path = index_path
        self.batch_size = batch_size

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
        self.model = FlagModel(config.model_name)
        self.index = faiss.IndexFlatIP(config.dim)
        self.dim = config.dim
        self.embeddings = []
        self.documents = []
        self.num_documents = 0
        self.index_path = config.index_path
        self.batch_size = config.batch_size

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
                os.makedirs(index_path, exist_ok=True)
                print(f"Index saving to：{index_path}")
            np.savez(
                os.path.join(index_path, 'document.vecstore'),
                embeddings=self.embeddings,
                documents=self.documents
            )
            faiss.write_index(self.index, os.path.join(index_path, 'fassis.index'))
            print("Index saved successfully to", index_path)

    def get_embedding(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings for a list of sentences."""
        return self.model.encode(sentences=sentences, batch_size=self.batch_size)  # Using configured batch_size

    def add_texts(self, texts: List[str]):
        """Add multiple texts to the index."""
        embeddings = self.get_embedding(texts)
        self.index.add(embeddings)
        self.documents.extend(texts)
        self.embeddings.extend(embeddings)
        self.num_documents += len(texts)

    def add_text(self, text: str):
        """Add a single text to the index."""
        self.add_texts([text])

    def build_from_texts(self, corpus: List[str]):
        """Process and index a list of texts in batches."""
        if not corpus:
            return

        for i in tqdm(range(0, len(corpus), self.batch_size), desc="Building index"):
            batch = corpus[i:i + self.batch_size]
            self.add_texts(batch)

    def retrieve(self, query: str = None, top_k: int = 5):
        D, I = self.index.search(self.get_embedding([query]), top_k)
        return [{'text': self.documents[idx], 'score': score} for idx, score in zip(I[0], D[0])]
