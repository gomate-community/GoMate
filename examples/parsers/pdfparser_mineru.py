from gomate.modules.document.pdf_mineru_parser import  PdfParserWithMinerU
if __name__ == '__main__':
    pdf_parser=PdfParserWithMinerU(url='http://localhost:8888/pdf_parse')
    pdf_file_path= 'H:/Projects/GoMate/data/docs/small_ocr.pdf'
    pdf_parser.parse(pdf_file_path)