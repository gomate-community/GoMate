import json
import os
import random
from typing import Any
from typing import List

import faiss
import numpy as np
from tqdm import tqdm

from gomate.modules.retrieval.base import BaseRetriever
from gomate.modules.retrieval.embedding import BaseEmbeddingModel, OpenAIEmbeddingModel
from gomate.modules.retrieval.embedding import SBertEmbeddingModel


class FaissRetrieverConfig:
    def __init__(
            self,
            embedding_model=None,
            question_embedding_model=None,
            embedding_model_string=None,
            index_path=None,
            rebuild_index=True
    ):

        if embedding_model is not None and not isinstance(
                embedding_model, BaseEmbeddingModel
        ):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel or None"
            )

        if question_embedding_model is not None and not isinstance(
                question_embedding_model, BaseEmbeddingModel
        ):
            raise ValueError(
                "question_embedding_model must be an instance of BaseEmbeddingModel or None"
            )

        self.embedding_model = embedding_model or OpenAIEmbeddingModel()
        self.question_embedding_model = question_embedding_model or self.embedding_model
        self.embedding_model_string = embedding_model_string or "OpenAI"
        self.index_path = index_path
        self.rebuild_index = rebuild_index

    def log_config(self):
        config_summary = """
            FaissRetrieverConfig:
                Embedding Model: {embedding_model}
                Question Embedding Model: {question_embedding_model}
                Embedding Model String: {embedding_model_string}
                Index Path: {index_path}
                Rebuild Index Path: {rebuild_index}""".format(
            embedding_model=self.embedding_model,
            question_embedding_model=self.question_embedding_model,
            embedding_model_string=self.embedding_model_string,
            index_path=self.index_path,
            rebuild_index=self.rebuild_index
        )
        return config_summary


class FaissRetriever(BaseRetriever):
    """
    FaissRetriever is a class that retrieves similar context chunks for a given query using Faiss.
    encoders_type is 'same' if the question and context encoder is the same,
    otherwise, encoders_type is 'different'.
    """

    def __init__(self, config):
        self.embedding_model = config.embedding_model
        self.question_embedding_model = config.question_embedding_model
        self.index = None
        self.documents = []
        self.embeddings = []
        self.embedding_model_string = config.embedding_model_string
        self.index_path = config.index_path
        self.rebuild_index = config.rebuild_index
        # Load the index from the specified path if it is not None
        if not self.rebuild_index:
            if self.index_path and os.path.exists(self.index_path):
                self.load_index(self.index_path)
        else:
            if os.path.exists(self.index_path):
                os.remove(self.index_path)

    def load_index(self, index_path):
        """
        Loads a Faiss index from a specified path.

        :param index_path: Path to the Faiss index file.
        """
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print("Index loaded successfully.")
        else:
            print("Index path does not exist.")

    def encode_document(self, doc_text):
        """
        Builds the index from a given text.

        :param doc_text: A string containing the document text.
        """
        embeddings = []
        embedding = self.embedding_model.create_embedding(doc_text)
        embeddings.append(embedding)

        embeddings = np.array(embeddings, dtype=np.float32)
        return embeddings, [doc_text]

    def build_from_texts(self, documents: List[str] = None):
        """
        Processes multiple documents in batches, builds the index, and saves it to disk.
        :param documents: List of document texts to process.
        """

        for i in tqdm(range(0, len(documents))):
            doc_embeddings, context_chunks = self.encode_document(documents[i])
            self.embeddings.append(doc_embeddings)
            self.documents.extend(context_chunks)
        # Initialize the index only once
        if self.index is None and self.embeddings:
            self.index = faiss.IndexFlatIP(self.embeddings[0].shape[1])

        self.embeddings = np.vstack(self.embeddings)
        self.index.add(self.embeddings)
        # Save the index to disk
        print(f"save index to:{self.index_path}")
        faiss.write_index(self.index, self.index_path)

    def sanity_check(self, num_samples=4):
        """
        Perform a sanity check by recomputing embeddings of a few randomly-selected chunks.

        :param num_samples: The number of samples to test.
        """
        indices = random.sample(range(len(self.documents)), num_samples)

        for i in indices:
            original_embedding = self.embeddings[i]
            recomputed_embedding = self.embedding_model.create_embedding(
                self.documents[i]
            )
            assert np.allclose(
                original_embedding, recomputed_embedding
            ), f"Embeddings do not match for index {i}!"

        print(f"Sanity check passed for {num_samples} random samples.")

    def retrieve(self, query: str = None, top_k: int = 5) -> list[Any]:
        """
        Retrieves the k most similar context chunks for a given query.

        :param query: A string containing the query.
        :param top_k: An integer representing the number of similar context chunks to retrieve.
        :return: A string containing the retrieved context chunks.
        """
        query_embedding = np.array(
            [
                np.array(
                    self.question_embedding_model.create_embedding(query),
                    dtype=np.float32,
                ).squeeze()
            ]
        )
        context = []
        distances, indices = self.index.search(query_embedding, top_k)
        for i in range(top_k):
            context.append({'text': self.documents[indices[0][i]], 'score': distances[0][i]})
        return context


if __name__ == '__main__':

    embedding_model_path = "/data/users/searchgpt/pretrained_models/bge-large-zh-v1.5"
    embedding_model = SBertEmbeddingModel(embedding_model_path)
    retriever_config = FaissRetrieverConfig(
        embedding_model=embedding_model,
        embedding_model_string="bge-large-zh-v1.5",
        index_path="/data/users/searchgpt/yq/GoMate/examples/retrievers/faiss_index.bin",
        rebuild_index=False
    )
    faiss_retriever = FaissRetriever(config=retriever_config)
    documents = []
    with open('/data/users/searchgpt/yq/GoMate/data/docs/zh_refine.json', 'r', encoding="utf-8") as f:
        for line in f.readlines():
            data = json.loads(line)
            documents.extend(data['positive'])
            documents.extend(data['negative'])
    faiss_retriever.build_from_texts(documents[:200])
    search_contexts = faiss_retriever.retrieve("2021年香港GDP增长了多少")
    print(search_contexts)
