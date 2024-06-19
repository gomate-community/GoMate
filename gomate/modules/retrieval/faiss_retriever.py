import json
import os
import shutil
from typing import Any

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from tqdm import tqdm
from langchain_core.embeddings import Embeddings

from gomate.modules.retrieval.base import BaseRetriever
from gomate.modules.retrieval.embedding import BaseEmbeddingModel, OpenAIEmbeddingModel
from gomate.modules.retrieval.embedding import SBertEmbedding,TextEmbedding


class FaissRetrieverConfig:
    def __init__(
            self,
            embedding_model=None,
            top_k=5,
            embedding_model_string=None,
            vectorstore_path=None,
            rebuild_index=True
    ):

        if embedding_model is not None and not isinstance(
                embedding_model, Embeddings
        ):
            raise ValueError(
                "embedding_model must be an instance of Embeddings or None"
            )

        self.top_k = top_k
        self.embedding_model = embedding_model or OpenAIEmbeddingModel()
        self.embedding_model_string = embedding_model_string or "OpenAI"
        self.vectorstore_path = vectorstore_path
        self.rebuild_index = rebuild_index

    def log_config(self):
        config_summary = """
		FaissRetrieverConfig:
			Embedding Model: {embedding_model}
			Top K: {top_k}
			Embedding Model String: {embedding_model_string}
			Index Path: {vectorstore_path}
			Rebuild Index Path: {rebuild_index}
		""".format(
            embedding_model=self.embedding_model,
            top_k=self.top_k,
            embedding_model_string=self.embedding_model_string,
            vectorstore_path=self.vectorstore_path,
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
        self.vector_store = None
        self.context_chunks = []
        self.top_k = config.top_k
        self.embedding_model_string = config.embedding_model_string
        self.vectorstore_path = config.vectorstore_path
        self.rebuild_index = config.rebuild_index
        # Load the index from the specified path if it is not None
        if not self.rebuild_index:
            if self.vectorstore_path and os.path.exists(self.vectorstore_path):
                self.load_index(self.vectorstore_path)
        else:
            if self.vectorstore_path and os.path.exists(self.vectorstore_path):
                shutil.rmtree(self.vectorstore_path)

    def load_index(self, vectorstore_path):
        """
        Loads a Faiss index from a specified path.

        :param vectorstore_path: Path to the Faiss index file.
        """
        if os.path.exists(vectorstore_path):
            self.vector_store = FAISS.load_local(
                vectorstore_path,
                embeddings=self.embedding_model,
                index_name='index',
                allow_dangerous_deserialization=True
            )

            print("Index loaded successfully.")
        else:
            print("Index path does not exist.")

    def build_from_documents(self, documents):
        """
        Processes multiple documents in batches, builds the index, and saves it to disk.

        :param documents: List of document texts to process.
        :param save_path: Path to save the index file.
        :param batch_size: Number of documents to process in each batch.
        """

        self.vector_store = FAISS.from_documents(documents, embedding=self.embedding_model)
        # for document in tqdm(documents, desc="build_from_documents"):
        #     self.vector_store.add_documents([document])
        #     del document
                # if self.vector_store:
                #     self.vector_store.add_documents([document])
                # else:
                #     self.vector_store = FAISS.from_documents([document], self.embedding_model)

        self.vector_store.save_local(self.vectorstore_path)

    def retrieve(self, query: str) -> list[Any]:
        """
        Retrieves the k most similar context chunks for a given query.

        :param query: A string containing the query.
        :param k: An integer representing the number of similar context chunks to retrieve.
        :return: A string containing the retrieved context chunks.
        """
        docs = self.vector_store.similarity_search_with_score(query)
        return docs


if __name__ == '__main__':

    embedding_model_path = "/home/test/pretrained_models/bge-large-zh-v1.5"
    embedding_model = TextEmbedding(embedding_model_path)
    retriever_config = FaissRetrieverConfig(
        embedding_model=embedding_model,
        top_k=5,
        embedding_model_string="bge-large-zh-v1.5",
        vectorstore_path="zh_refine_index",
        rebuild_index=True
    )

    faiss_retriever = FaissRetriever(config=retriever_config)

    documents = []
    with open('/home/test/codes/GoMate/data/zh_refine.json', 'r', encoding="utf-8") as f:
        for line in f.readlines():
            data = json.loads(line)
            for article in data['positive']:
                documents.append(
                    Document(page_content=article)
                )
            for article in data['negative']:
                documents.append(
                    Document(page_content=article)
                )
    faiss_retriever.build_from_documents(documents[:100])
    contexts = faiss_retriever.retrieve("首趟RCEP班列的起点")
    print(contexts)
