#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: views.py
@time: 2024/06/13
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import loguru
from fastapi import APIRouter

from api.apps.core.judge.bodys import JudgeBody
from api.apps.handle.response.json_response import ApiResponse
from gomate.modules.judger.bge_judger import BgeJudger, BgeJudgerConfig
from gomate.modules.judger.chatgpt_judger import OpenaiJudger, OpenaiJudgerConfig

judge_router = APIRouter()

judge_config = BgeJudgerConfig(
    model_name_or_path="/data/users/searchgpt/pretrained_models/bge-reranker-large"
)
bge_judger = BgeJudger(judge_config)

judger_config = OpenaiJudgerConfig(
    api_url="https://aicloud.oneainexus.cn:30013/inference/aicloud-yanqiang/gomatellm/"
)
openai_judger = OpenaiJudger(judger_config)


# Create
@judge_router.post("/judge/", response_model=None, summary="判断文档相关性")
async def judge(judge_body: JudgeBody):
    contexts = judge_body.contexts
    query = judge_body.query
    method = judge_body.method

    if method == 'llm':
        loguru.logger.info("llm judge ...")
        judge_docs = openai_judger.judge(
            query=query,
            documents=contexts,
            is_sorted=False
        )
    else:
        loguru.logger.info("bge judge...")
        judge_docs = bge_judger.judge(
            query=query,
            documents=contexts,
            is_sorted=False
        )
    return ApiResponse(judge_docs, message="判断文档是否相关成功")
