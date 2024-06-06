#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/5/31 1:23
"""
from gomate.modules.rewriter.base import BaseRewriter

import numpy as np
import time
import openai
import cohere

WEB_SEARCH = """Please write a passage to answer the question.
Question: {}
Passage:"""

SCIFACT = """Please write a scientific paper passage to support/refute the claim.
Claim: {}
Passage:"""

ARGUANA = """Please write a counter argument for the passage.
Passage: {}
Counter Argument:"""

TREC_COVID = """Please write a scientific paper passage to answer the question.
Question: {}
Passage:"""

FIQA = """Please write a financial article passage to answer the question.
Question: {}
Passage:"""

DBPEDIA_ENTITY = """Please write a passage to answer the question.
Question: {}
Passage:"""

TREC_NEWS = """Please write a news passage about the topic.
Topic: {}
Passage:"""

MR_TYDI = """Please write a passage in {} to answer the question in detail.
Question: {}
Passage:"""


class Promptor:
    def __init__(self, task: str, language: str = 'en'):
        self.task = task
        self.language = language

    def build_prompt(self, query: str):
        if self.task == 'web search':
            return WEB_SEARCH.format(query)
        elif self.task == 'scifact':
            return SCIFACT.format(query)
        elif self.task == 'arguana':
            return ARGUANA.format(query)
        elif self.task == 'trec-covid':
            return TREC_COVID.format(query)
        elif self.task == 'fiqa':
            return FIQA.format(query)
        elif self.task == 'dbpedia-entity':
            return DBPEDIA_ENTITY.format(query)
        elif self.task == 'trec-news':
            return TREC_NEWS.format(query)
        elif self.task == 'mr-tydi':
            return MR_TYDI.format(self.language, query)
        else:
            raise ValueError('Task not supported')

class Generator:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key

    def generate(self):
        return ""


class OpenAIGenerator(Generator):
    def __init__(self, model_name, api_key, n=8, max_tokens=512, temperature=0.7, top_p=1, frequency_penalty=0.0,
                 presence_penalty=0.0, stop=['\n\n\n'], wait_till_success=False):
        super().__init__(model_name, api_key)
        self.n = n
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.wait_till_success = wait_till_success

    @staticmethod
    def parse_response(response):
        to_return = []
        for _, g in enumerate(response['choices']):
            text = g['text']
            logprob = sum(g['logprobs']['token_logprobs'])
            to_return.append((text, logprob))
        texts = [r[0] for r in sorted(to_return, key=lambda tup: tup[1], reverse=True)]
        return texts

    def generate(self, prompt):
        get_results = False
        while not get_results:
            try:
                result = openai.Completion.create(
                    engine=self.model_name,
                    prompt=prompt,
                    api_key=self.api_key,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    top_p=self.top_p,
                    n=self.n,
                    stop=self.stop,
                    logprobs=1
                )
                get_results = True
            except Exception as e:
                if self.wait_till_success:
                    time.sleep(1)
                else:
                    raise e
        return self.parse_response(result)


class CohereGenerator(Generator):
    def __init__(self, model_name, api_key, n=8, max_tokens=512, temperature=0.7, p=1, frequency_penalty=0.0,
                 presence_penalty=0.0, stop=['\n\n\n'], wait_till_success=False):
        super().__init__(model_name, api_key)
        self.cohere = cohere.Cohere(self.api_key)
        self.n = n
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.p = p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.wait_till_success = wait_till_success

    @staticmethod
    def parse_response(response):
        text = response.generations[0].text
        return text

    def generate(self, prompt):
        texts = []
        for _ in range(self.n):
            get_result = False
            while not get_result:
                try:
                    result = self.cohere.generate(
                        prompt=prompt,
                        model=self.model_name,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        frequency_penalty=self.frequency_penalty,
                        presence_penalty=self.presence_penalty,
                        p=self.p,
                        k=0,
                        stop=self.stop,
                    )
                    get_result = True
                except Exception as e:
                    if self.wait_till_success:
                        time.sleep(1)
                    else:
                        raise e
            text = self.parse_response(result)
            texts.append(text)
        return texts

class HyDE:
    def __init__(self, promptor, generator, encoder, searcher):
        self.promptor = promptor
        self.generator = generator
        self.encoder = encoder
        self.searcher = searcher

    def prompt(self, query):
        return self.promptor.build_prompt(query)

    def generate(self, query):
        prompt = self.promptor.build_prompt(query)
        hypothesis_documents = self.generator.generate(prompt)
        return hypothesis_documents

    def encode(self, query, hypothesis_documents):
        all_emb_c = []
        for c in [query] + hypothesis_documents:
            c_emb = self.encoder.encode(c)
            all_emb_c.append(np.array(c_emb))
        all_emb_c = np.array(all_emb_c)
        avg_emb_c = np.mean(all_emb_c, axis=0)
        hyde_vector = avg_emb_c.reshape((1, len(avg_emb_c)))
        return hyde_vector

    def search(self, hyde_vector, k=10):
        hits = self.searcher.search(hyde_vector, k=k)
        return hits

    def e2e_search(self, query, k=10):
        prompt = self.promptor.build_prompt(query)
        hypothesis_documents = self.generator.generate(prompt)
        hyde_vector = self.encode(query, hypothesis_documents)
        hits = self.searcher.search(hyde_vector, k=k)
        return hits

class HydeRewriter(BaseRewriter):
    def __init__(self):
        pass