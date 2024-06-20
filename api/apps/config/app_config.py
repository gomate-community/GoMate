#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: app_config.py.py
@time: 2024/06/13
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import pprint
from typing import ClassVar

pp = pprint.PrettyPrinter(indent=4)


class AppConfig:
    """配置类"""
    API_V1_STR: str = ""
    # 文档接口描述相关的配置
    DOCS_URL = API_V1_STR + '/docs'
    REDOC_URL = API_V1_STR + '/redocs'
    OPENAPI_URL = API_V1_STR + '/openapi_url'
    API_VERSION = "v1"

    TITLE = "FASTAPI 模板函数"

    DESC = """

           """
    TAGS_METADATA = [
    ]
    # 配置代理相关的参数信息
    SERVERS = [
        {"url": "/", "description": "开发接口地址"},
        {"url": "/v2", "description": "测试地址"},
    ]

    WEB_URL: ClassVar[str] = '*'
    # 接口地址
    API_URL: ClassVar[str] = 'http://127.0.0.1:10001'
    # 运行访问的地址
    API_HOST: ClassVar[str] = '0.0.0.0'
    # 端口
    API_PORT: int = 10000

    DEBUGGER: bool = True

    SHOW_DOCS: bool = True
