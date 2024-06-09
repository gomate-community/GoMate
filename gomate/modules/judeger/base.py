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

    def __init__(self, config):
        self.config = config
        self.name = config['judger_name']
        self.device = config['device']

    def run(self, item) -> str:
        """Get judgement result.

        Args:
            item: dataset item, contains question, retrieval result...

        Returns:
            judgement: bool, whether to retreive
        """
        pass

    def batch_run(self, dataset, batch_size=None) -> List[str]:
        return [self.run(item) for item in dataset]
