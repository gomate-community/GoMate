import os
from gomate.modules.document.pdf_mineru_parser import  PdfParserWithMinerU
from gomate.modules.document.utils import PROJECT_BASE
from tqdm import tqdm
if __name__ == '__main__':
    pdf_parser=PdfParserWithMinerU(url='http://localhost:8888/pdf_parse')
    pdf__path= f'{PROJECT_BASE}/data/docs/计算所现行制度汇编202406/documents'
    for filename in tqdm(os.listdir(pdf__path)):
        if filename.endswith('.pdf'):
            pdf_file_path=f'{pdf__path}/{filename}'
            pdf_parser.parse(pdf_file_path)