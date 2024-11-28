from trustrag.modules.document.common_parser import CommonParser

if __name__ == '__main__':
    parser = CommonParser()
    document_path = '/data/users/searchgpt/yq/GoMate_dev/docs/夏至各地习俗.docx'
    chunks = parser.parse(document_path)
    print(chunks)
