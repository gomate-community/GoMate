#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: textparser.py
@time: 2024/06/06
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import typing
from typing import Optional

from langchain_core.documents.base import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from gomate.modules.document.utils import contains_text
from gomate.modules.document.base import BaseParser


class TextParser(BaseParser):
    """
    TextParser is a parser class for processing plain text files.
    """

    supported_file_extensions = [".txt"]

    def __init__(self, max_chunk_size: int = 1000, *args, **kwargs):
        """
        Initializes the TextParser object.
        """
        self.max_chunk_size = max_chunk_size

    def get_chunks(
        self, filepath: str,  *args, **kwargs
    ) -> typing.List[Document]:
        """
        Asynchronously loads the text from a text file and returns it in chunks.
        """
        content = None
        with open(filepath, "r",encoding="utf-8") as f:
            content = f.read()
        if not content:
            print("Error reading file: " + filepath)
            return []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.max_chunk_size)
        texts = text_splitter.split_text(content)

        docs = [
            Document(
                page_content=text,
                metadata={
                    "type": "text",
                },
            )
            for text in texts
            if contains_text(text)
        ]

        return docs