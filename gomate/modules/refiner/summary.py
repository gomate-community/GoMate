#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: summary.py
@time: 2024/05/30
@contact: yanqiangmiffy@gamil.com
@description: summary models
"""
import logging
from abc import ABC, abstractmethod

import torch
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass


class GLMSummarizationModel(BaseSummarizationModel):
    def __init__(self, model_name_or_path: str = '') -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.load_model()

    @retry(wait=wait_random_exponential(min=60, max=6000), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):
        prompt = f"Write a summary of the following, including as many key details as possible: {context}:"
        response, history = self.model.chat(self.tokenizer, prompt, None)
        return response

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, torch_dtype=torch.float16,
                                                          trust_remote_code=True).cuda()


class GPT3TurboSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-3.5-turbo"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


class GPT3SummarizationModel(BaseSummarizationModel):
    def __init__(self, model="text-davinci-003"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


if __name__ == '__main__':
    glm_summarization = GLMSummarizationModel(model_name_or_path=r"I:\pretrained_models\llm\chatglm3-6b")
    with open(r'H:\Projects\GoMate\data\docs\sample.txt', 'r') as file:
        text = file.read()

    summarization_text=glm_summarization.summarize(text[:200])
    print(summarization_text)