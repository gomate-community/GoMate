#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: processor.py.py
@time: 2024/06/13
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import traceback

from fastapi import FastAPI, Request
from pydantic.v1 import IntegerError, MissingError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.exceptions import HTTPException as FastapiHTTPException
from fastapi.exceptions import RequestValidationError
from api.apps.handle.response.json_response import *

#  这个可能有点问题
DEBUGGER=True


class ApiExceptionHandler():
    def __init__(self, app=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if app is not None:
            self.init_app(app)

    def init_app(self, app: FastAPI):
        app.add_exception_handler(Exception, handler=self.all_exception_handler)
        # 捕获StarletteHTTPException返回的错误异常，如返回405的异常的时候，走的是这个地方
        app.add_exception_handler(StarletteHTTPException, handler=self.http_exception_handler)
        app.add_exception_handler(RequestValidationError, handler=self.validation_exception_handler)
        app.add_exception_handler(TypeError, handler=self.all_exception_handler)

    async def validation_exception_handler(self, request: Request, exc: RequestValidationError):
        # print("参数提交异常错误selfself", exc.errors()[0].get('loc'))
        # 路径参数错误
        # 判断错误类型
        if isinstance(exc.raw_errors[0].exc, IntegerError):
            pass
        elif isinstance(exc.raw_errors[0].exc, MissingError):
            pass
        return ParameterResponse(code=400, message='参数校验错误', data={
            "detail": exc.errors(),
            "body": exc.body
        })

    async def all_exception_handler(self, request: Request, exc: Exception):

        if isinstance(exc, StarletteHTTPException) or isinstance(exc, FastapiHTTPException):
            if exc.status_code == 405:
                return MethodnotallowedResponse()
            if exc.status_code == 404:
                return NotfoundResponse()
            elif exc.status_code == 429:
                return LimiterResResponse()
            elif exc.status_code == 500:
                detail = exc.detail
                if not DEBUGGER:
                    return CommonResponse(code=500, msg=detail)
                else:
                    traceback.print_exc()
                    return InternalErrorResponse()
            elif exc.status_code == 400:
                # 有部分的地方直接的选择使用raise的方式抛出了异常，这里也需要进程处理
                # raise HTTPException(HTTP_400_BAD_REQUEST, 'Invalid token')
                return BadRequestResponse(msg=exc.detail)

            return BadRequestResponse()
        else:
            # 其他内部的异常的错误拦截处理
            # logger.exception(exc)

            if False:
                return InternalErrorResponse()
            else:
                traceback.print_exc()
                return InternalErrorResponse()

    async def http_exception_handler(self, request: Request, exc: StarletteHTTPException):
        '''
           全局的捕获抛出的HTTPException异常，注意这里需要使用StarletteHTTPException的才可以
           :param request:
           :param exc:
           :return:
           '''
        if exc.status_code == 405:
            return MethodnotallowedResponse()
        if exc.status_code == 404:
            return NotfoundResponse()
        elif exc.status_code == 429:
            return LimiterResResponse()
        elif exc.status_code == 500:
            detail = exc.detail
            if not DEBUGGER:
                return CommonResponse(code=500, msg=detail)
            else:
                traceback.print_exc()
                return InternalErrorResponse()
        elif exc.status_code == 400:
            # 有部分的地方直接的选择使用raise的方式抛出了异常，这里也需要进程处理
            # raise HTTPException(HTTP_400_BAD_REQUEST, 'Invalid token')
            return BadRequestResponse(msg=exc.detail)
