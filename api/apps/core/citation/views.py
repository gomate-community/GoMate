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
from trustrag.modules.citation.match_citation import MatchCitation
from trustrag.modules.citation.source_citation import SourceCitation
mc = MatchCitation()
sc = SourceCitation()
citation_router = APIRouter()


# Create
@citation_router.post("/citation/", response_model=None, summary="答案引用")
async def citation(citation_body: CitationBody):
    question = citation_body.question
    response = citation_body.response
    evidences = citation_body.evidences
    selected_idx = citation_body.selected_idx
    show_code = citation_body.show_code
    show_summary = citation_body.show_summary
    selected_docs=citation_body.selected_docs
    # loguru.logger.info(response)
    loguru.logger.info(show_summary)
    print(show_summary)
    try:
        show_summary=True
        if not show_summary:
            citation_response = mc.ground_response(
                question=question,
                response=response,
                evidences=evidences,
                selected_idx=selected_idx,
                markdown=True,
                show_code=show_code,
                selected_docs=selected_docs
            )
        else:
            citation_response = sc.ground_response(
                question=question,
                response=response,
                evidences=evidences,
                selected_idx=selected_idx,
                markdown=True,
                show_code=show_code,
                selected_docs=selected_docs
            )
    except:
        loguru.logger.error("引文引用报错，使用生成式引用")
        citation_response = mc.ground_response(
            question=question,
            response=response,
            evidences=evidences,
            selected_idx=selected_idx,
            markdown=True,
            show_code=show_code,
            selected_docs=selected_docs
        )
    # loguru.logger.info(citation_response)
    return ApiResponse(citation_response, message="答案引用成功")
