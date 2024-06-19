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
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from .handle.exception.processor import ApiExceptionHandler
from .config.app_config import AppConfig

settings = AppConfig()


class FastSkeletonApp():
    # 定义FastAPIapp实例的对象
    fastapi = FastAPI(
        title=settings.TITLE,
        description=settings.DESC,
        version=settings.API_VERSION,
        debug=False,
        docs_url=settings.DOCS_URL,
        openapi_url=settings.OPENAPI_URL,
        redoc_url=settings.REDOC_URL,
        openapi_tags=settings.TAGS_METADATA,
        servers=settings.SERVERS,
    )

    def __init__(self):
        self.register_global_exception()

        self.fastapi.add_middleware(
            CORSMiddleware,
            allow_origins=[settings.WEB_URL],  # 允许进行跨域请求的来源列表，*作为通配符
            allow_credentials=True,  # 跨域请求支持cookie，默认为否
            allow_methods=["*"],  # 允许跨域请求的HTTP方法
            allow_headers=["*"],  # 允许跨域请求的HTTP头列表
        )

    def register_global_exception(self):
        ApiExceptionHandler().init_app(self.fastapi)

    def register_global_cors(self):
        self.fastapi.add_middleware(
            CORSMiddleware,
            allow_origins=[settings.WEB_URL],  # 允许进行跨域请求的来源列表，*作为通配符
            allow_credentials=True,  # 跨域请求支持cookie，默认为否
            allow_methods=["*"],  # 允许跨域请求的HTTP方法
            allow_headers=["*"],  # 允许跨域请求的HTTP头列表
        )


def create_app():
    app = FastSkeletonApp()
    # 返回fastapi的App对象
    return app.fastapi
