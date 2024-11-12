#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: bodys.py
@time: 2024/06/13
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
from typing import List

from pydantic import BaseModel, Field


class RewriteBody(BaseModel):
    """
    # 入参模型定义
    """
    query: str = Field("RCEP具体包括哪些国家", description="查询query")
