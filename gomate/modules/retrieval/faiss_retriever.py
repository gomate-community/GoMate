import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from typing import List, Any

import faiss
import numpy as np
import tiktoken
from tqdm import tqdm

from gomate.modules.retrieval.embedding import BaseEmbeddingModel, OpenAIEmbeddingModel
from gomate.modules.retrieval.embedding import SBertEmbeddingModel
from gomate.modules.retrieval.retrievers import BaseRetriever
from gomate.modules.retrieval.utils import split_text


class FaissRetrieverConfig:
    def __init__(
            self,
            max_tokens=100,
            max_context_tokens=3500,
            use_top_k=True,
            embedding_model=None,
            question_embedding_model=None,
            top_k=5,
            tokenizer=None,
            embedding_model_string=None,
            index_path=None,
            rebuild_index=True
    ):
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if max_context_tokens is not None and max_context_tokens < 1:
            raise ValueError("max_context_tokens must be at least 1 or None")

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

        self.top_k = top_k
        self.max_tokens = max_tokens
        self.max_context_tokens = max_context_tokens
        self.use_top_k = use_top_k
        self.embedding_model = embedding_model or OpenAIEmbeddingModel()
        self.question_embedding_model = question_embedding_model or self.embedding_model
        self.tokenizer = tokenizer
        self.embedding_model_string = embedding_model_string or "OpenAI"
        self.index_path = index_path
        self.rebuild_index=rebuild_index

    def log_config(self):
        config_summary = """
		FaissRetrieverConfig:
			Max Tokens: {max_tokens}
			Max Context Tokens: {max_context_tokens}
			Use Top K: {use_top_k}
			Embedding Model: {embedding_model}
			Question Embedding Model: {question_embedding_model}
			Top K: {top_k}
			Tokenizer: {tokenizer}
			Embedding Model String: {embedding_model_string}
			Index Path: {index_path}
			Rebuild Index Path: {rebuild_index}
		""".format(
            max_tokens=self.max_tokens,
            max_context_tokens=self.max_context_tokens,
            use_top_k=self.use_top_k,
            embedding_model=self.embedding_model,
            question_embedding_model=self.question_embedding_model,
            top_k=self.top_k,
            tokenizer=self.tokenizer,
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
        self.context_chunks = []
        self.max_tokens = config.max_tokens
        self.max_context_tokens = config.max_context_tokens
        self.use_top_k = config.use_top_k
        self.tokenizer = config.tokenizer
        self.top_k = config.top_k
        self.embedding_model_string = config.embedding_model_string
        self.index_path = config.index_path
        self.rebuild_index=config.rebuild_index
        # Load the index from the specified path if it is not None
        if not self.rebuild_index:
            if self.index_path and os.path.exists(self.index_path):
                self.load_index(self.index_path)
        else:
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
        # Split the text into context chunks
        context_chunks = np.array(
            split_text(doc_text, self.tokenizer, self.max_tokens)
        )
        # Collect embeddings using a for loop
        embeddings = []
        for context_chunk in context_chunks:
            embedding = self.embedding_model.create_embedding(context_chunk)
            embeddings.append(embedding)

        embeddings = np.array(embeddings, dtype=np.float32)
        return embeddings,context_chunks.tolist()
    def build_from_texts(self, documents):
        """
        Processes multiple documents in batches, builds the index, and saves it to disk.

        :param documents: List of document texts to process.
        :param save_path: Path to save the index file.
        :param batch_size: Number of documents to process in each batch.
        """
        self.all_embeddings = []
        self.context_chunks=[]
        for i in tqdm(range(0, len(documents))):
            doc_embeddings,context_chunks = self.encode_document(documents[i])
            self.all_embeddings.append(doc_embeddings)
            self.context_chunks.extend(context_chunks)
        # Initialize the index only once
        if self.index is None and self.all_embeddings:
            self.index = faiss.IndexFlatIP(self.all_embeddings[0].shape[1])

        # first_shape = self.all_embeddings[0].shape
        # for embedding in self.all_embeddings:
        #     if embedding.shape != first_shape:
        #         print("Found an embedding with a different shape:", embedding.shape)

        self.all_embeddings = np.vstack(self.all_embeddings)
        print(self.all_embeddings.shape)
        print(len(self.context_chunks))
        self.index.add(self.all_embeddings)
        # Save the index to disk
        faiss.write_index(self.index, self.index_path)
    def sanity_check(self, num_samples=4):
        """
        Perform a sanity check by recomputing embeddings of a few randomly-selected chunks.

        :param num_samples: The number of samples to test.
        """
        indices = random.sample(range(len(self.context_chunks)), num_samples)

        for i in indices:
            original_embedding = self.all_embeddings[i]
            recomputed_embedding = self.embedding_model.create_embedding(
                self.context_chunks[i]
            )
            assert np.allclose(
                original_embedding, recomputed_embedding
            ), f"Embeddings do not match for index {i}!"

        print(f"Sanity check passed for {num_samples} random samples.")

    def retrieve(self, query: str) -> list[Any]:
        """
        Retrieves the k most similar context chunks for a given query.

        :param query: A string containing the query.
        :param k: An integer representing the number of similar context chunks to retrieve.
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

        if self.use_top_k:
            distances, indices = self.index.search(query_embedding, self.top_k)
            print(distances,indices)
            print(distances[0][2],indices)
            for i in range(self.top_k):
                context.append({'text':self.context_chunks[indices[0][i]],'score':distances[0][i]})
        else:
            range_ = int(self.max_context_tokens / self.max_tokens)
            _, indices = self.index.search(query_embedding, range_)
            total_tokens = 0
            for i in range(range_):
                tokens = len(self.tokenizer.encode(self.context_chunks[indices[0][i]]))
                context.append(self.context_chunks[indices[0][i]])
                if total_tokens + tokens > self.max_context_tokens:
                    break
                total_tokens += tokens

        return context


if __name__ == '__main__':
    from transformers import AutoTokenizer

    embedding_model_path = "/home/test/pretrained_models/bge-large-zh-v1.5"
    embedding_model = SBertEmbeddingModel(embedding_model_path)
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
    retriever_config = FaissRetrieverConfig(
        max_tokens=100,
        max_context_tokens=3500,
        use_top_k=True,
        embedding_model=embedding_model,
        top_k=5,
        tokenizer=tokenizer,
        embedding_model_string="bge-large-zh-v1.5",
        index_path="faiss_index.bin",
        rebuild_index=True
    )

    faiss_retriever=FaissRetriever(config=retriever_config)

    documents=[]
    with open('/home/test/codes/GoMate/data/zh_refine.json','r',encoding="utf-8") as f:
        for line in f.readlines():
            data=json.loads(line)
            documents.extend(data['positive'])
            documents.extend(data['negative'])
    print(len(documents))
    faiss_retriever.build_from_texts(documents[:200])

    contexts=faiss_retriever.retrieve("2022年冬奥会开幕式总导演是谁")
    print(contexts)