#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yanqiangmiffy
@contact:1185918903@qq.com
@license: Apache Licence
@time: 2024/6/11 11:16
"""
from datasets import load_dataset

dataset = load_dataset("Tevatron/wikipedia-nq-corpus")

dataset.save_to_disk(r"H:\Projects\GoMate\data\docs\wikipedia-nq-corpus")

