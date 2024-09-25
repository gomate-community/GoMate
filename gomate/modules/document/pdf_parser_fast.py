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
from io import BytesIO
from PyPDF2 import PdfReader as pdf2_read
import fitz
from tqdm import tqdm
import logging

class PdfParserUsingPyMuPDF():
    """
    PdfParserUsingPyMuPDF is a parser class for extracting text from PDF files using PyMuPDF library.
    """

    supported_file_extensions = [".pdf"]

    def __init__(self, max_chunk_size: int = 1000, *args, **kwargs):
        """
        Initializes the PdfParserUsingPyMuPDF object.
        """
        self.max_chunk_size = max_chunk_size

    def parse(
            self, fnm: str, *args, **kwargs
    ):
        """
        Asynchronously extracts text from a PDF file and returns it in chunks.
        """
        final_texts = []
        final_tables = []
        # Open the PDF file using pdfplumber
        doc = fitz.open(fnm) if isinstance(
            fnm, str) else fitz.open(
            BytesIO(fnm))
        for page in tqdm(doc, total=len(doc), desc="get pages"):
            table = page.find_tables()
            table = list(table)
            for ix, tab in enumerate(table):
                tab = tab.extract()
                tab = list(map(lambda x: [str(t) for t in x], tab))
                tab = list(map("||".join, tab))
                tab = "\n".join(tab)
                final_tables.append(tab)

            text = page.get_text()
            # print(text)
            # clean up text for any problematic characters
            text = re.sub("\n", " ", text).strip()
            # text = text.encode("ascii", errors="ignore").decode("ascii")
            # text = re.sub(r"([^\w\s])\1{4,}", " ", text)
            # text = re.sub(" +", " ", text).strip()

            final_texts.append(text)

        return final_texts + final_tables

class PdfSimParser(object):
    def parse(self, filename, from_page=0, to_page=100000, **kwargs):
        self.outlines = []
        lines = []
        try:
            self.pdf = pdf2_read(
                filename if isinstance(
                    filename, str) else BytesIO(filename))
            for page in self.pdf.pages[from_page:to_page]:
                lines.extend([t for t in page.extract_text().split("\n")])

            outlines = self.pdf.outline

            def dfs(arr, depth):
                for a in arr:
                    if isinstance(a, dict):
                        self.outlines.append((a["/Title"], depth))
                        continue
                    dfs(a, depth + 1)

            dfs(outlines, 0)
        except Exception as e:
            logging.warning(f"Outlines exception: {e}")
        if not self.outlines:
            logging.warning(f"Miss outlines")

        return lines

    def crop(self, ck, need_position):
        raise NotImplementedError

    @staticmethod
    def remove_tag(txt):
        raise NotImplementedError

if __name__ == '__main__':
    pdf_parser = PdfParserUsingPyMuPDF()
    contents = pdf_parser.parse('/data/users/searchgpt/yq/GoMate_dev/data/docs/新冠肺炎疫情.pdf')
    print(contents)
