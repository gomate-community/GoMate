#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: json_response.py.py
@time: 2024/06/13
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import time
from datetime import datetime
from typing import Any, Dict, Optional,List
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic.deprecated.json import ENCODERS_BY_TYPE


def custom_datetime_encoder(date_obj):
    return int(date_obj.timestamp())


class ApiResponse(JSONResponse, Exception):
    # 定义返回响应码--如果不指定的话则默认都是返回200
    data: Optional[Dict[str, Any]] = []  # 结果可以是{} 或 []
    message = '成功'
    timestamp = int(time.time() * 1000)
    api_code = 200

    def __init__(self, data: object = None, code: object = None, message: object = None, **options: object) -> None:
        if data is None:
            data = []
        if message:
            self.message = message
        if data:
            self.data = data
        if data == "":  # 如果data是空字符串，那么就返回空列表
            self.data = data
        if code:
            self.api_code = code
        body = dict(
            message=self.message,
            code=self.api_code,
            data=self.data,
            timestamp=self.timestamp,
        )
        #  定制化json返回的格式
        ENCODERS_BY_TYPE[datetime] = custom_datetime_encoder
        super(ApiResponse, self).__init__(status_code=200, content=jsonable_encoder(body), **options)

class CommonResponse(ApiResponse):
    def __init__(self, api_code=500, message="出现异常", **options):
        super().__init__(code=api_code, message=message, **options)

    result = []


class BadRequestResponse(ApiResponse):
    api_code = 10000
    result = []
    message = '缺少用户信息'

# ParameterResponse
class ParameterResponse(ApiResponse):
    api_code = 10001
    result = []
    message = '参数校验错误'
#MethodnotallowedResponse
class MethodnotallowedResponse(ApiResponse):
    api_code = 10002
    result = []
    message = '请求方法错误'

#NotfoundResponse
class NotfoundResponse(ApiResponse):
    api_code = 10003
    result = []
    message = '请求路径错误'

class LimiterResResponse(ApiResponse):
    api_code = 10004
    result = []
    message = '请求过于频繁，请稍后再试'

class InternalErrorResponse(ApiResponse):
    api_code = 10005
    result = []
    message = '服务器内部错误'

#  用户找不到
class UserNotFoundResponse(ApiResponse):
    api_code = 10006
    result = []
    message = '用户不存在'
