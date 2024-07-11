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

judge_router = APIRouter()
judge_config = BgeJudgerConfig(
    model_name_or_path="/data/users/searchgpt/pretrained_models/bge-reranker-large"
)
bge_judger = BgeJudger(judge_config)


# Create
@judge_router.post("/judge/", response_model=None, summary="判断文档相关性")
async def judge(judge_body: JudgeBody):
    contexts = judge_body.contexts
    query = judge_body.query
    loguru.logger.info(query)
    loguru.logger.info(contexts)
    judge_docs = bge_judger.judge(
        query=query,
        documents=contexts,
        is_sorted=False
    )
    loguru.logger.info(judge_docs)
    return ApiResponse(judge_docs, message="判断文档是否相关成功")
