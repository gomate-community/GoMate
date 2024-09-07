#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: pdfparser_example.py
@time: 2024/06/06
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
from gomate.modules.document.pdf_parser_fast import PdfParserUsingPyMuPDF


if __name__ == '__main__':
    parser=PdfParserUsingPyMuPDF(max_chunk_size=1000)
    # chunks=parser.parse(fnm="../../data/docs/汽车操作手册.pdf")
    # print(len(chunks))
    # for chunk in chunks:
    #     print(chunk)

    chunks = parser.parse(fnm="../../data/docs/计算所现行制度汇编202406/计算所现行制度汇编202406/综合处/中国科学院计算技术研究所综合安全管理制度_20240531修订版.pdf")
    print(chunks)
    print(len(chunks))
    for chunk in chunks:
        print(chunk)