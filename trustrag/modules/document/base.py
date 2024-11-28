#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: parser.py
@time: 2024/05/24
@contact: yanqiangmiffy@gamil.com
"""
import typing
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional
from langchain.docstore.document import Document


class BaseParser(ABC):
    """
    BaseParser is an Abstract Base Class (ABC) that serves as a template for all parser objects.
    It contains the common attributes and methods that each parser should implement.
    """

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    async def get_chunks(
        self,
        filepath: str,
        metadata: Optional[dict],
        *args,
        **kwargs,
    ) -> typing.List[Document]:
        """
        Abstract method. This should asynchronously read a file and return its content in chunks.

        Parameters:
            loaded_data_point (LoadedDataPoint): Loaded Document to read and parse.

        Returns:
            typing.List[Document]: A list of Document objects, each representing a chunk of the file.
        """
        pass