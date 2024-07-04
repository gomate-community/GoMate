#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: models.py
@time: 2024/06/13
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
from pydantic import BaseModel
from pydantic import Field
from enum import IntEnum
from tortoise import fields
from apps.core.base import AbstractModel

# class ApplicationModel(BaseModel):
#     application_name: str = Field("应用名称", description="应用名称")
#     knowledge_id: str = Field("应用名称", description="知识库结合id")
#     service_name: str = Field("官方体验服务", description="模型服务名称")
#     model_name: str = Field("GoGPT2.0", description="大模型名称")
#     temperature: float = Field(0.5, description="多样性大小")


class Application(AbstractModel):
    application_name = fields.CharField(max_length=150, description="应用名称")
    knowledge_id = fields.IntField(default=1, description="知识库id")
    # knowledge_name=fields.CharField(max_length=150,description="知识库名称")
    # service_name = fields.CharField(max_length=150, description="模型服务名称")
    # model_name = fields.CharField(max_length=150, description="大模型名称")
    service_id=fields.IntField(default=1,description="模型服务id")
    model_id=fields.IntField(default=1,description="模型id")
    temperature = fields.FloatField(default=0.5, description="多样性大小")

    class Meta:
        table = "rag_application"

