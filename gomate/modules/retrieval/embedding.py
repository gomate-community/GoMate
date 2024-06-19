#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: embedding.py
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
"""
import logging
import os
from typing import List
import logging
from abc import ABC, abstractmethod
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)



class TextEmbedding(Embeddings, ABC):
    def __init__(self, emb_model_name_or_path, batch_size=64, max_len=512, device='cuda', **kwargs):

        super().__init__(**kwargs)
        self.model = AutoModel.from_pretrained(emb_model_name_or_path, trust_remote_code=True).half().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(emb_model_name_or_path, trust_remote_code=True)
        if 'bge' in emb_model_name_or_path:
            self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："
        else:
            self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH = ""
        self.emb_model_name_or_path = emb_model_name_or_path
        self.device = device
        self.batch_size = batch_size
        self.max_len = max_len
        print("successful load embedding model")

    def compute_kernel_bias(self, vecs, n_components=384):
        """
            bertWhitening: https://spaces.ac.cn/archives/8069
            计算kernel和bias
            vecs.shape = [num_samples, embedding_size]，
            最后的变换：y = (x + bias).dot(kernel)
        """
        mu = vecs.mean(axis=0, keepdims=True)
        cov = np.cov(vecs.T)
        u, s, vh = np.linalg.svd(cov)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))
        return W[:, :n_components], -mu

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
            Compute corpus embeddings using a HuggingFace transformer model.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        num_texts = len(texts)
        texts = [t.replace("\n", " ") for t in texts]
        sentence_embeddings = []

        for start in tqdm(range(0, num_texts, self.batch_size),desc="embed_documents"):
            end = min(start + self.batch_size, num_texts)
            batch_texts = texts[start:end]
            encoded_input = self.tokenizer(batch_texts, max_length=512, padding=True, truncation=True,
                                           return_tensors='pt').to(self.device)

            with torch.no_grad():

                model_output = self.model(**encoded_input)
                # Perform pooling. In this case, cls pooling.
                if 'gte' in self.emb_model_name_or_path:
                    batch_embeddings = model_output.last_hidden_state[:, 0]
                else:
                    batch_embeddings = model_output[0][:, 0]

                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                sentence_embeddings.extend(batch_embeddings.tolist())

        # sentence_embeddings = np.array(sentence_embeddings)
        # self.W, self.mu = self.compute_kernel_bias(sentence_embeddings)
        # sentence_embeddings = (sentence_embeddings+self.mu) @ self.W
        # self.W, self.mu = torch.from_numpy(self.W).cuda(), torch.from_numpy(self.mu).cuda()
        return sentence_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
            Compute query embeddings using a HuggingFace transformer model.
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        if 'bge' in self.emb_model_name_or_path:
            encoded_input = self.tokenizer([self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH + text], padding=True,
                                           truncation=True, return_tensors='pt').to(self.device)
        else:
            encoded_input = self.tokenizer([text], padding=True,
                                           truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        # sentence_embeddings = (sentence_embeddings + self.mu) @ self.W
        return sentence_embeddings[0].tolist()

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        pass


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-ada-002"):
        self.client = OpenAI()
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )



class SBertEmbedding(Embeddings,ABC):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text,show_progress_bar=False)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings=self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        embeddings = self.model.encode(text, show_progress_bar=False)
        print(embeddings)
        print(embeddings.tolist())
        print(embeddings.tolist()[0])
        print(embeddings.tolist()[0])
        return embeddings.tolist()

class BaseEmbeddings:
    """
    Base class for embeddings
    """

    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api

    def get_embedding(self, text: str, model: str) -> List[float]:
        raise NotImplementedError

    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude


class OpenAIEmbedding(BaseEmbeddings):
    """
    class for OpenAI embeddings
    """

    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from openai import OpenAI
            self.client = OpenAI()
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            self.client.base_url = os.getenv("OPENAI_BASE_URL")

    def get_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        if self.is_api:
            text = text.replace("\n", " ")
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding
        else:
            raise NotImplementedError


class JinaEmbedding(BaseEmbeddings):
    """
    class for Jina embeddings
    """

    def __init__(self, path: str = 'jinaai/jina-embeddings-v2-base-zh', is_api: bool = False) -> None:
        super().__init__(path, is_api)
        self._model = self.load_model()

    def get_embedding(self, text: str) -> List[float]:
        return self._model.encode([text])[0].tolist()

    def load_model(self):
        import torch
        from transformers import AutoModel
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = AutoModel.from_pretrained(self.path, trust_remote_code=True).to(device)
        return model


class ZhipuEmbedding(BaseEmbeddings):
    """
    class for Zhipu embeddings
    """

    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from zhipuai import ZhipuAI
            self.client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model="embedding-2",
            input=text,
        )
        return response.data[0].embedding


class DashscopeEmbedding(BaseEmbeddings):
    """
    class for Dashscope embeddings
    """

    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            import dashscope
            dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
            self.client = dashscope.TextEmbedding

    def get_embedding(self, text: str, model: str = 'text-embedding-v1') -> List[float]:
        response = self.client.call(
            model=model,
            input=text
        )
        return response.output['embeddings'][0]['embedding']


class BgeEmbedding(BaseEmbeddings):
    """
    class for BGE embeddings
    """

    def __init__(self, path: str = 'BAAI/bge-base-zh-v1.5', is_api: bool = False) -> None:
        super().__init__(path, is_api)
        self._model, self._tokenizer = self.load_model(path)

    def get_embedding(self, text: str) -> List[float]:
        import torch
        encoded_input = self._tokenizer([text], padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.to(self._model.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self._model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings[0].tolist()

    def load_model(self, path: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(path).to(device)
        model.eval()
        return model, tokenizer
