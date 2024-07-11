#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: reranker.py
@time: 2024/05/22
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
from abc import ABC
from typing import List


class BaseJudger(ABC):
    """Base object of Judger, used for judging whether to retrieve"""

    def judge(self, text: str, content: List[str]) -> List[str]:
        raise NotImplementedError
