#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: pdfparser_fast.py
@time: 2024/06/06
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import re
from typing import Optional

import fitz
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from gomate.modules.document.utils import contains_text
from gomate.modules.document.base import BaseParser
from tqdm import tqdm
class PdfParserUsingPyMuPDF(BaseParser):
    """
    PdfParserUsingPyMuPDF is a parser class for extracting text from PDF files using PyMuPDF library.
    """

    supported_file_extensions = [".pdf"]

    def __init__(self, max_chunk_size: int = 1000, *args, **kwargs):
        """
        Initializes the PdfParserUsingPyMuPDF object.
        """
        self.max_chunk_size = max_chunk_size

    def get_chunks(
        self, filepath: str, *args, **kwargs
    ):
        """
        Asynchronously extracts text from a PDF file and returns it in chunks.
        """
        final_texts = []
        final_tables = []
        try:
            # Open the PDF file using pdfplumber
            doc = fitz.open(filepath)
            for page in tqdm(doc,total=len(doc),desc="get pages"):
                table = page.find_tables()
                table = list(table)
                for ix, tab in enumerate(table):
                    tab = tab.extract()
                    tab = list(map(lambda x: [str(t) for t in x], tab))
                    tab = list(map("||".join, tab))
                    tab = "\n".join(tab)
                    tab = [
                        Document(
                            page_content=tab,
                            metadata={
                                "page_num": page.number,
                                "type": "table",
                                "table_num": ix,
                            },
                        )
                    ]
                    final_tables.extend(tab)

                text = page.get_text()

                # clean up text for any problematic characters
                text = re.sub("\n", " ", text).strip()
                text = text.encode("ascii", errors="ignore").decode("ascii")
                text = re.sub(r"([^\w\s])\1{4,}", " ", text)
                text = re.sub(" +", " ", text).strip()

                # Create a Document object per page with page-specific metadata
                if len(text) > self.max_chunk_size:
                    # Split the text into chunks of size less than or equal to max_chunk_size
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.max_chunk_size, chunk_overlap=0
                    )
                    text_splits = text_splitter.split_text(text)
                    texts = [
                        Document(
                            page_content=text_split,
                            metadata={
                                "page_num": page.number,
                                "type": "text",
                            },
                        )
                        for text_split in text_splits
                        if contains_text(text_split)
                    ]
                    final_texts.extend(texts)
                else:
                    if contains_text(text):
                        final_texts.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "page_num": page.number,
                                    "type": "text",
                                },
                            )
                        )
        except Exception:
            print(f"Error while parsing PDF file at {filepath}")
            # Return an empty list if there was an error during processing
            return []

        return final_texts + final_tables