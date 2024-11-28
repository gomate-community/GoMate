import requests
from io import BytesIO

# # 创建BytesIO对象
# data = BytesIO("我喜欢中国。北京特别好玩。".encode('utf-8'))
# data = BytesIO("Hello!\nHi!\nGoodbye!".encode('utf-8'))
# # 准备文件数据
# files = {'file': ('testfile.txt', data)}
#
# # 发送POST请求
# response = requests.post("http://127.0.0.1:10000/gomate_tool/parse/", files=files)
#
# # 打印响应内容
# print(response.json())


import requests

# 指定文件路径
file_path = '/data/users/searchgpt/yq/GoMate/data/docs/夏至各地习俗.docx'
files = {'file': open(file_path, 'rb')}

# 发送文件
response = requests.post("http://127.0.0.1:10000/gomate_tool/parse/", files=files)
print(response.json())

files['file'].close()
