#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: markdown_parser.py
@time: 2024/06/06
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
from gomate.modules.document.markdown_parser import MarkdownParser


if __name__ == '__main__':
    markdown_parser=MarkdownParser(max_chunk_size=100)

    chunks=markdown_parser.get_chunks(filepath="../../data/docs/bm25算法.md")

    print(len(chunks))

    for chunk in chunks:
        print(chunk.page_content)