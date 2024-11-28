#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/5/31 0:31
"""
from abc import ABC, abstractmethod

class BaseRewriter(ABC):
    @abstractmethod
    def rewrite(self, query: str) -> str:
        pass
