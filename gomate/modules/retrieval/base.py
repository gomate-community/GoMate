#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: retrievers.py
@time: 2024/05/30
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str,top_k:int) -> str:
        pass
