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
from fastapi import APIRouter

from api.apps.core.rerank.bodys import RerankBody
from api.apps.core.rerank.models import Application
from api.apps.handle.response.json_response import UserNotFoundResponse, ApiResponse
from gomate.modules.reranker.bge_reranker import BgeReranker, BgeRerankerConfig

# from apps.handle.exception.exception import MallException
# from apps.core.config.models import LLMModel
# from tortoise.contrib.pydantic import pydantic_model_creator

rerank_router = APIRouter()
reranker_config = BgeRerankerConfig(
    model_name_or_path="/data/users/searchgpt/pretrained_models/bge-reranker-large"
)
bge_reranker = BgeReranker(reranker_config)
# Create
@rerank_router.post("/rerank/", response_model=None, summary="重排序检索文档")
async def rerank(rerank_body: RerankBody):
    contexts = rerank_body.contexts
    top_docs = bge_reranker.rerank(
        query=rerank_body.query,
        documents=contexts,
    )
    return ApiResponse(top_docs, message="创建应用成功")
