#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/7/2 23:32
https://github.com/NLPJCL/RAG-Retrieval/blob/master/rag_retrieval/infer/reranker_models/llm_rankers.py
"""
from typing import Union, List, Optional, Tuple
from .ranker import BaseRanker
from .result import RankedResults, Result
from .utils import get_device, get_dtype, vprint

import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LLMRanker(BaseRanker):

    def __init__(self,
                 model_name_or_path: str,
                 dtype: str = None,
                 device: str = None,
                 verbose: int = 1,
                 ):

        self.verbose = verbose
        self.model_name_or_path = model_name_or_path
        self.device = get_device(device, verbose=self.verbose)
        self.dtype = get_dtype(dtype, device=self.device, verbose=self.verbose)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        model_dtype = next(self.model.parameters()).dtype
        vprint(f"Loaded model {self.model_name_or_path}", self.verbose)
        vprint(f"model_dtype is  {model_dtype}", self.verbose)

        self.model.eval()

        self.yes_loc = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][0]

    @torch.no_grad()
    def compute_score(self,
                      sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
                      batch_size: int = 16,
                      max_length: int = 1024,
                      normalize: bool = False,
                      prompt: str = None,
                      cutoff_layers: list = None,
                      enable_tqdm: bool = True,
                      ):

        all_scores = []
        for start_index in tqdm.tqdm(range(0, len(sentence_pairs), batch_size), desc="Compute Scores",
                                     disable=not enable_tqdm):

            sentences_batch = sentence_pairs[start_index:start_index + batch_size]
            inputs = self.get_inputs(sentences_batch, prompt, max_length)
            inputs = inputs.to(self.device)

            if 'layerwise' in self.model_name_or_path:
                logits = self.model(**inputs, return_dict=True, cutoff_layers=cutoff_layers).logits
                # [batch_size,seq_len]
                scores = logits[0][:, -1].cpu().float().tolist()
            else:
                logits = self.model(**inputs, return_dict=True).logits
                # [batch_size,seq_len,vocabulary_dim]
                scores = logits[:, -1, self.yes_loc].view(-1, ).cpu().float().tolist()
            all_scores.extend(scores)

        if normalize == True:
            all_scores = [sigmoid(score) for score in all_scores]

        if len(all_scores) == 1:
            return all_scores[0]

        return all_scores

    @torch.no_grad()
    def rerank(self,
               query: str,
               docs: Union[List[str], str] = None,
               batch_size: int = 256,
               normalize: bool = False,
               prompt: str = None,
               cutoff_layers: list = None,
               max_length: int = 1024,
               long_doc_process_strategy: str = "max_score_slice",  # ['max_score_slice','max_length_truncation']
               ):
        # remove invalid docs
        docs = [doc[:128000] for doc in docs if isinstance(doc, str) and 0 < len(doc)]

        if query is None or len(query) == 0 or len(docs) == 0:
            return {'rerank_docs': [], 'rerank_scores': []}

        vprint(f'long_doc_process_strategy is {long_doc_process_strategy}', self.verbose)
        if long_doc_process_strategy == 'max_length_truncation':
            return self.__max_length_truncation_rerank(query, docs, batch_size, max_length, normalize, prompt,
                                                       cutoff_layers)
        else:
            return self.__max_score_slice_rerank(query, docs, batch_size, max_length, normalize, prompt, cutoff_layers)

    @torch.no_grad()
    def __max_length_truncation_rerank(self,
                                       query: str,
                                       docs: Union[List[str], str] = None,
                                       batch_size: int = 32,
                                       max_length: int = 1024,
                                       normalize: bool = False,
                                       prompt: str = None,
                                       cutoff_layers: list = None,
                                       ):
        doc_ids = list(range(len(docs)))
        sentence_pairs = [[query, doc] for doc in docs]
        all_scores = self.compute_score(
            sentence_pairs,
            batch_size=batch_size,
            max_length=max_length,
            normalize=normalize,
            prompt=prompt,
            cutoff_layers=cutoff_layers,
            enable_tqdm=False
        )
        ranked_results = [
            Result(doc_id=doc_id, text=doc, score=score, rank=idx + 1)
            for idx, (doc_id, doc, score) in enumerate(
                sorted(zip(doc_ids, docs, all_scores), key=lambda x: x[2], reverse=True)
            )
        ]
        return RankedResults(results=ranked_results, query=query, has_scores=True)

    @torch.no_grad()
    def __max_score_slice_rerank(self,
                                 query: str,
                                 docs: Union[List[str], str] = None,
                                 batch_size: int = 32,
                                 max_length: int = 1024,
                                 normalize: bool = False,
                                 prompt: str = None,
                                 cutoff_layers: list = None,
                                 overlap_tokens_length: int = 80,
                                 ):

        doc_ids = list(range(len(docs)))

        # preproc of tokenization
        sentence_pairs, sentence_pairs_idxs = self.__reranker_tokenize_preproc(
            query,
            docs,
            max_length=max_length,
            prompt=prompt,
            overlap_tokens_length=overlap_tokens_length,
        )

        sentence_pairs = sentence_pairs.to(self.device)
        # batch inference
        # if self.num_gpus > 1:
        #     batch_size = batch_size * self.num_gpus

        all_scores = []
        for start_index in range(0, len(sentence_pairs), batch_size):
            inputs = sentence_pairs[start_index:start_index + batch_size]

            if 'layerwise' in self.model_name_or_path:
                logits = self.model(**inputs, return_dict=True, cutoff_layers=cutoff_layers).logits
                scores = logits[0][:, -1].cpu().float().tolist()
            else:
                logits = self.model(**inputs, return_dict=True).logits
                scores = logits[:, -1, self.yes_loc].view(-1, ).cpu().float().tolist()
            all_scores.extend(scores)

        # ranking
        merge_scores = [float("-inf") for _ in range(len(docs))]
        for idx, score in zip(sentence_pairs_idxs, all_scores):
            merge_scores[idx] = max(merge_scores[idx], score)

        if normalize == True:
            merge_scores = [sigmoid(score) for score in merge_scores]

        ranked_results = [
            Result(doc_id=doc_id, text=doc, score=score, rank=idx + 1)
            for idx, (doc_id, doc, score) in enumerate(
                sorted(zip(doc_ids, docs, merge_scores), key=lambda x: x[2], reverse=True)
            )
        ]
        return RankedResults(results=ranked_results, query=query, has_scores=True)

    def __reranker_tokenize_preproc(self,
                                    query: str,
                                    docs: List[str],
                                    max_length: int = 1024,
                                    prompt: str = None,
                                    overlap_tokens_length: int = 80,
                                    ):

        if prompt is None:
            prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
        sep = "\n"
        prompt_inputs = self.tokenizer(prompt,
                                       return_tensors=None,
                                       add_special_tokens=False)['input_ids']
        sep_inputs = self.tokenizer(sep,
                                    return_tensors=None,
                                    add_special_tokens=False)['input_ids']

        query_inputs = self.tokenizer(f'A: {query}',
                                      return_tensors=None,
                                      add_special_tokens=False,
                                      max_length=max_length * 4 // 5,
                                      truncation=True)['input_ids']

        max_doc_inputs_length = max_length - len(query_inputs) - 2
        overlap_tokens_length_implt = min(overlap_tokens_length, max_doc_inputs_length // 4)

        sentence_pairs = []
        sentence_pairs_idxs = []

        for idx, doc in enumerate(docs):

            doc_inputs = self.tokenizer(f'B: {doc}',
                                        return_tensors=None,
                                        add_special_tokens=False,
                                        max_length=max_length,
                                        truncation=True)

            doc_inputs_length = len(doc_inputs['input_ids'])

            if doc_inputs_length <= max_doc_inputs_length:

                item = self.tokenizer.prepare_for_model(
                    [self.tokenizer.bos_token_id] + query_inputs,
                    sep_inputs + doc_inputs['input_ids'],
                    truncation='only_second',
                    max_length=max_length,
                    padding=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    add_special_tokens=False
                )
                item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
                item['attention_mask'] = [1] * len(item['input_ids'])
                sentence_pairs.append(item)
                sentence_pairs_idxs.append(idx)
            else:
                start_id = 0
                while start_id < doc_inputs_length:
                    end_id = start_id + max_doc_inputs_length
                    sub_doc_inputs = {k: v[start_id:end_id] for k, v in doc_inputs.items()}
                    start_id = end_id - overlap_tokens_length_implt if end_id < doc_inputs_length else end_id

                    item = self.tokenizer.prepare_for_model(
                        [self.tokenizer.bos_token_id] + query_inputs,
                        sep_inputs + sub_doc_inputs['input_ids'],
                        truncation='only_second',
                        max_length=max_length,
                        padding=False,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                        add_special_tokens=False
                    )
                    item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
                    item['attention_mask'] = [1] * len(item['input_ids'])
                    sentence_pairs.append(item)
                    sentence_pairs_idxs.append(idx)

        sentence_pairs = self.tokenizer.pad(
            sentence_pairs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors='pt',
        )
        return sentence_pairs, sentence_pairs_idxs

    def get_inputs(self,
                   pairs,
                   prompt=None,
                   max_length=1024):
        if prompt is None:
            prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
        sep = "\n"
        prompt_inputs = self.tokenizer(prompt,
                                       return_tensors=None,
                                       add_special_tokens=False)['input_ids']
        sep_inputs = self.tokenizer(sep,
                                    return_tensors=None,
                                    add_special_tokens=False)['input_ids']
        inputs = []
        for query, passage in pairs:
            query_inputs = self.tokenizer(f'A: {query}',
                                          return_tensors=None,
                                          add_special_tokens=False,
                                          max_length=max_length * 3 // 4,
                                          truncation=True)
            passage_inputs = self.tokenizer(f'B: {passage}',
                                            return_tensors=None,
                                            add_special_tokens=False,
                                            max_length=max_length,
                                            truncation=True)
            item = self.tokenizer.prepare_for_model(
                [self.tokenizer.bos_token_id] + query_inputs['input_ids'],
                sep_inputs + passage_inputs['input_ids'],
                truncation='only_second',
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False
            )
            item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
            item['attention_mask'] = [1] * len(item['input_ids'])
            inputs.append(item)
        return self.tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors='pt',
        )

# pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]

# model_name_or_path='./bge-reranker-v2-minicpm-layerwise/models--BAAI--bge-reranker-v2-minicpm-layerwise/snapshots/47b5332b296c4d8cb6ee2c60502cc62a0d708881'
# reranker=llmreranker(model_name_or_path,dtype='fp16')

# scores = reranker.compute_score(pairs)
# print(scores)

# query='what is panda?'

# docs=['hi','The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']

# scores = reranker.rerank(query,docs)
# print(scores)