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

from api.apps.core.refiner.bodys import RefinerBody
from api.apps.handle.response.json_response import ApiResponse
from trustrag.modules.refiner.compressor import LLMCompressApi

refiner_router = APIRouter()
compressor = LLMCompressApi()


# Create
@refiner_router.post("/refiner/", response_model=None, summary="压缩文档")
async def refiner(refiner_body: RefinerBody):
    contexts = refiner_body.contexts
    query = refiner_body.query
    # loguru.logger.info(query)
    # loguru.logger.info(contexts)
    response = compressor.compress(query, contexts)
    data = {'compress_content': response['response']}
    loguru.logger.info(data)
    return ApiResponse(data, message="重构文档成功")
