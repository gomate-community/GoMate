# GoMate

可配置的模块化RAG框架。

[![Python](https://img.shields.io/badge/Python-3.10.0-3776AB.svg?style=flat)](https://www.python.org)
![workflow status](https://github.com/gomate-community/rageval/actions/workflows/makefile.yml/badge.svg)
[![codecov](https://codecov.io/gh/gomate-community/GoMate/graph/badge.svg?token=eG99uSM8mC)](https://codecov.io/gh/gomate-community/GoMate)
[![pydocstyle](https://img.shields.io/badge/pydocstyle-enabled-AD4CD3)](http://www.pydocstyle.org/en/stable/)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)


## 🔥Gomate 简介
GoMate是一款配置化模块化的Retrieval-Augmented Generation (RAG) 框架，旨在提供**可靠的输入与可信的输出**，确保用户在检索问答场景中能够获得高质量且可信赖的结果。

GoMate框架的设计核心在于其**高度的可配置性和模块化**，使得用户可以根据具体需求灵活调整和优化各个组件，以满足各种应用场景的要求。

## 🔨Gomate框架
![framework.png](resources%2Fframework.png)
## ✨主要特色

**“Reliable input,Trusted output”**

可靠的输入，可信的输出

## 🚀快速上手

### 安装环境
```shell
pip install -r requirements.txt
```
### 导入模块
```python
from gomate.modules.document.reader import ReadFiles
from gomate.modules.generator.llm import GLMChat
from gomate.modules.retrieval.embedding import BgeEmbedding
from gomate.modules.store.vector import VectorStore
```
### 文档解析
```python
docs = ReadFiles('./data/docs').get_content(max_token_len=600, cover_content=150)
vector = VectorStore(docs)
```

### 提取向量

```python
embedding = BgeEmbedding("BAAI/bge-large-zh-v1.5")  # 创建EmbeddingModel
vector.get_vector(EmbeddingModel=embedding)
vector.persist(path='storage')  # 将向量和文档内容保存到storage目录下，下次再用就可以直接加载本地的数据库
vector.load_vector(path='storage')  # 加载本地的数据库
```

### 检索文档

```python
question = '伊朗坠机事故原因是什么？'
contents = vector.query(question, EmbeddingModel=embedding, k=1)
content = '\n'.join(contents[:5])
print(contents)
```

### 大模型问答
```python
chat = GLMChat(path='THUDM/chatglm3-6b')
print(chat.chat(question, [], content))
```

### 添加文档
```python
docs = ReadFiles('').get_content_by_file(file='data/add/伊朗问题.txt', max_token_len=600, cover_content=150)
vector.add_documents('storage', docs, embedding)
question = '如今伊朗人的经济生活状况如何？'
contents = vector.query(question, EmbeddingModel=embedding, k=1)
content = '\n'.join(contents[:5])
print(contents)
print(chat.chat(question, [], content))
```

## 🔧定制化RAG

> 构建自定义的RAG应用

```python
from gomate.modules.document.reader import ReadFiles
from gomate.modules.generator.llm import GLMChat
from gomate.modules.retrieval.embedding import BgeEmbedding
from gomate.modules.store.vector import VectorStore

class RagApplication():
    def __init__(self, config):
       pass
    def init_vector_store(self):
       pass
    def load_vector_store(self):
        pass
    def add_document(self, file_path):
        pass

    def chat(self, question: str = '', topk: int = 5):
        pass
```

模块可见[rag.py](gomate/applications/rag.py)


### 🌐体验RAG效果
可以配置本地模型路径
```text
class ApplicationConfig:
    llm_model_name = '/data/users/searchgpt/pretrained_models/chatglm3-6b'  # 本地模型文件 or huggingface远程仓库
    embedding_model_name = '/data/users/searchgpt/pretrained_models/bge-reranker-large'  # 检索模型文件 or huggingface远程仓库
    vector_store_path = './storage'
    docs_path = './data/docs'

```

```shell
python app.py
```
浏览器访问：[127.0.0.1:7860](127.0.0.1:7860)
![demo.png](resources%2Fdemo.png)