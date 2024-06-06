#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: utils.py
@time: 2024/06/06
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
def contains_text(text):
    # Check if the token contains at least one alphanumeric character
    return any(char.isalnum() for char in text)