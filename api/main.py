#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: main.py
@time: 2024/06/13
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import sys
sys.path.append('.')
sys.path.append('/data/users/searchgpt/yq/GoMate')
from apps.app import create_app
from apps.config.app_config import AppConfig
from apps.core.rerank.views import rerank_router
from apps.core.parser.views import parse_router
import uvicorn

app_config = AppConfig()
app = create_app()
app.include_router(rerank_router, prefix="/gomate_tool",tags=["rerank"])
app.include_router(parse_router, prefix="/gomate_tool",tags=["parser"])

if __name__ == '__main__':
    uvicorn.run('main:app', host=app_config.API_HOST, port=app_config.API_PORT,workers=8,reload=True)
