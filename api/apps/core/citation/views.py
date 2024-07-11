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

from api.apps.core.citation.bodys import CitationBody
from api.apps.handle.response.json_response import ApiResponse
from gomate.modules.citation.match_citation import MatchCitation

mc = MatchCitation()
citation_router = APIRouter()


# Create
@citation_router.post("/citation/", response_model=None, summary="答案引用")
async def citation(citation_body: CitationBody):
    response = citation_body.response
    evidences = citation_body.evidences
    selected_idx=citation_body.selected_idx
    # loguru.logger.info(response)
    # loguru.logger.info(evidences)
    citation_response = mc.ground_response(
        response=response,
        evidences=evidences,
        selected_idx=selected_idx,
        markdown=True
    )
    loguru.logger.info(citation_response)
    return ApiResponse(citation_response, message="答案引用成功")
