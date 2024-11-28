import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from pymilvus import connections, Collection, utility
from typing import List


class MilvusRetrieverConfig:
    def __init__(
            self,
            model_name_or_path='sentence-transformers/all-mpnet-base-v2',
            dim=768,
            collection_name='dense_retriever',
            host='localhost',
            port=19530
    ):
        self.model_name = model_name_or_path
        self.dim = dim
        self.collection_name = collection_name
        self.host = host
        self.port = port

    def log_config(self):
        config_summary = f"""
        MilvusRetrieverConfig:
        Model Name: {self.model_name}
        Dimension: {self.dim}
        Collection Name: {self.collection_name}
        Milvus Host: {self.host}
        Milvus Port: {self.port}
        """
        return config_summary


class MilvusRetriever:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name)

        # Connect to Milvus
        connections.connect(host=config.host, port=config.port)

        # Create or get the collection
        self._create_or_get_collection()

    def _create_or_get_collection(self):
        if utility.has_collection(self.config.collection_name):
            self.collection = Collection(self.config.collection_name)
        else:
            from pymilvus import FieldSchema, CollectionSchema, DataType
            fields = [
                FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=self.config.dim)
            ]
            schema = CollectionSchema(fields, self.config.collection_name)
            self.collection = Collection(self.config.collection_name, schema)

            # Create an IVF_FLAT index for the collection
            index_params = {
                'metric_type': 'L2',
                'index_type': 'IVF_FLAT',
                'params': {"nlist": 2048}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.detach().numpy()

    def add_text(self, text):
        embedding = self.get_embedding([text])[0]
        self.collection.insert([
            [text],
            [embedding.tolist()]
        ])

    def build_from_texts(self, texts: List[str]):
        embeddings = self.get_embedding(texts)
        entities = [
            [texts],
            embeddings.tolist()
        ]
        self.collection.insert(entities)

    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = self.get_embedding([query])[0]
        self.collection.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"]
        )

        return [{'text': hit.entity.get('text'), 'score': hit.distance} for hit in results[0]]

    def __del__(self):
        connections.disconnect(self.config.host)