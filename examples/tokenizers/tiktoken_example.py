#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/6/1 10:34
"""
import tiktoken

tokenizer=tiktoken.get_encoding("cl100k_base")
print(tokenizer)
print(tokenizer.encode("北京天安门"))
print(tokenizer.decode(tokenizer.encode("北京天安门")))