#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: textparser_exmaple.py
@time: 2024/06/06
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
from trustrag.modules.document.text_parser import TextParser




if __name__ == '__main__':
    text_parser=TextParser(
        max_chunk_size=512
    )

    # chunks=text_parser.get_chunks(
    #     filepath="../../data/docs/制度汇编.txt"
    # )
    chunks = text_parser.get_chunks(
        filepath="H:/2024-Xfyun-RAG/data/corpus.txt/corpus.txt"
    )
    print(len(chunks))

    for chunk in chunks:
        print(chunk)