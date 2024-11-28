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
from typing import List
from abc import ABC, abstractmethod


class BaseReranker(ABC):
    """
    Base class for reranker
    """
    def rerank(self, text: str, content: List[str], k: int) -> List[str]:
        raise NotImplementedError


