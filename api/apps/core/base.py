#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: base.py
@time: 2024/06/13
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
from datetime import datetime
from enum import IntEnum

from tortoise import models, fields


# 创建类型
class CreateType(IntEnum):
    CREATE = 0
    Fork = 1




class AbstractModel(models.Model):
    id = fields.IntField(pk=True)
    created_at = fields.DatetimeField(auto_now=True, description="创建时间")
    updated_at = fields.DatetimeField(null=True, description="更新时间")
    created_user_id = fields.IntField(default=0, description="创建人id")

    async def _pre_save(self, *args, **kwargs):
        self.updated_at = datetime.now()
        await super()._pre_save(*args, **kwargs)

    class Meta:
        # 抽象模型，不生成表
        abstract = True

