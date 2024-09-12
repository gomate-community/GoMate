import requests

class PdfParserWithMinerU:
    def __init__(self,url='http://localhost:8888/pdf_parse'):

        # 服务器URL
        self.url = url
    def parse(self,pdf_file_path):

        # PDF文件路径
        # pdf_file_path = 'path/to/your/file.pdf'

        # 请求参数
        params = {
            'parse_method': 'auto',
            'is_json_md_dump': 'true',
            'output_dir': 'output'
        }

        # 准备文件
        files = {
            'pdf_file': (pdf_file_path.split('/')[-1], open(pdf_file_path, 'rb'), 'application/pdf')
        }

        # 发送POST请求
        response = requests.post(self.url, params=params, files=files)

        # 检查响应
        if response.status_code == 200:
            print("PDF解析成功")
            print(response.json())
            print(response.json().keys())
        else:
            print(f"错误: {response.status_code}")
            print(response.text)


if __name__ == '__main__':
    pdf_parser=PdfParserWithMinerU(url='http://localhost:8888/pdf_parse')
    pdf_file_path= '/data/paper/16400599.pdf'
    pdf_parser.parse(pdf_file_path)