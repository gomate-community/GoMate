#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: llm.py
@time: 2024/05/16
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import os
from typing import Dict, List, Any

import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPT_TEMPLATE = dict(
    RAG_PROMPT_TEMPALTE="""使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:""",
    GoGPT_PROMPT_TEMPALTE="""请基于所提供的支撑信息和对话历史，对给定的问题撰写一个全面且有条理的答复。
    如果支撑信息或对话历史与当前问题无关或者提供信息不充分，请尝试自己回答问题或者无法回答问题。\n\n
    对话历史：{context}\n\n
    支撑信息：{concated_contents}\n\n
    问题：{query}\n\n回答：:""",
    InternLM_PROMPT_TEMPALTE="""先对上下文进行内容总结,再使用上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:""",
    GLM_PROMPT_TEMPALTE="""请结合参考的上下文内容回答用户问题，如果上下文不能支撑用户问题，那么回答不知道或者我无法根据参考信息回答。
            问题: {question}
            可参考的上下文：
            ···
            {context}
            ···
            如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
            有用的回答:"""
)


class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass


class OpenAIChat(BaseModel):
    def __init__(self, path: str = '', model: str = "gpt-3.5-turbo-1106") -> None:
        super().__init__(path)
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        client = OpenAI()
        client.api_key = os.getenv("OPENAI_API_KEY")
        client.base_url = os.getenv("OPENAI_BASE_URL")
        history.append({'role': 'user',
                        'content': PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt, context=content)})
        response = client.chat.completions.create(
            model=self.model,
            messages=history,
            max_tokens=150,
            temperature=0.1
        )
        return response.choices[0].message.content


class InternLMChat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def chat(self, prompt: str, history: List = [], content: str = '') -> str:
        prompt = PROMPT_TEMPLATE['InternLM_PROMPT_TEMPALTE'].format(question=prompt, context=content)
        response, history = self.model.chat(self.tokenizer, prompt, history)
        return response

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.float16,
                                                          trust_remote_code=True).cuda()


class GLMChat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def chat(self, prompt: str, history: List = [], content: str = '', llm_only: bool = False) -> tuple[Any, Any]:
        if llm_only:
            prompt = prompt
        else:
            prompt = PROMPT_TEMPLATE['GLM_PROMPT_TEMPALTE'].format(question=prompt, context=content)
        response, history = self.model.chat(self.tokenizer, prompt, history,max_length= 32000, num_beams=1, do_sample=True, top_p=0.8, temperature=0.2,)
        return response, history

    def load_model(self):

        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.float16,
                                                          trust_remote_code=True).cuda()

class QwenChat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()
        self.device='cuda'
    def chat(self, prompt: str, history: List = [], content: str = '', llm_only: bool = False) -> tuple[Any, Any]:
        if llm_only:
            prompt = prompt
        else:
            prompt = PROMPT_TEMPLATE['GLM_PROMPT_TEMPALTE'].format(question=prompt, context=content)
        messages = [
            {"role": "system", "content": "你是一个人工智能助手"},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response, history

    def load_model(self):

        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.float16,
        #                                                   trust_remote_code=True).cuda()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
class DashscopeChat(BaseModel):
    def __init__(self, path: str = '', model: str = "qwen-turbo") -> None:
        super().__init__(path)
        self.model = model

    def chat(self, prompt: str, history: List[Dict], content: str) -> str:
        import dashscope
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        history.append({'role': 'user',
                        'content': PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt, context=content)})
        response = dashscope.Generation.call(
            model=self.model,
            messages=history,
            result_format='message',
            max_tokens=150,
            temperature=0.1
        )
        return response.output.choices[0].message.content


class ZhipuChat(BaseModel):
    def __init__(self, path: str = '', model: str = "glm-4") -> None:
        super().__init__(path)
        from zhipuai import ZhipuAI
        self.client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))
        self.model = model

    def chat(self, prompt: str, history: List[Dict], content: str) -> str:
        history.append({'role': 'user',
                        'content': PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt, context=content)})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=history,
            max_tokens=150,
            temperature=0.1
        )
        return response.choices[0].message
