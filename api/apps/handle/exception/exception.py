#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: exception.py.py
@time: 2024/06/13
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
from api.apps.handle.response.json_response import ApiResponse


class MallException(Exception):
    def __init__(self, ApiResponse: ApiResponse):
        super().__init__(self)  # 初始化父类
        self.ApiResponse = ApiResponse

    def get_response(self):
        return self.ApiResponse

    def __str__(self):
        return self.ApiResponse.message
