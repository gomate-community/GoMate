# GoMate

可配置的模块化RAG框架。

[![Python](https://img.shields.io/badge/Python-3.10.0-3776AB.svg?style=flat)](https://www.python.org)
![workflow status](https://github.com/gomate-community/rageval/actions/workflows/makefile.yml/badge.svg)
[![codecov](https://codecov.io/gh/gomate-community/GoMate/graph/badge.svg?token=eG99uSM8mC)](https://codecov.io/gh/gomate-community/GoMate)
[![pydocstyle](https://img.shields.io/badge/pydocstyle-enabled-AD4CD3)](http://www.pydocstyle.org/en/stable/)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

## 🔥Gomate 简介

GoMate是一款配置化模块化的Retrieval-Augmented Generation (RAG) 框架，旨在提供**可靠的输入与可信的输出**
，确保用户在检索问答场景中能够获得高质量且可信赖的结果。

GoMate框架的设计核心在于其**高度的可配置性和模块化**，使得用户可以根据具体需求灵活调整和优化各个组件，以满足各种应用场景的要求。

## 🔨Gomate框架

![framework.png](resources%2Fframework.png)

## ✨主要特色

**“Reliable input,Trusted output”**

可靠的输入，可信的输出

## 🏗️ 更新记录

- 添加[MinerU文档解析](https://github.com/gomate-community/GoMate/blob/main/docs/mineru.md)
  ：一站式开源高质量数据提取工具，支持PDF/网页/多格式电子书提取`[20240907] `
- RAPTOR:递归树检索器实现
- 支持多种文件解析并且模块化目前支持解析的文件类型包括：`text`,`docx`,`ppt`,`excel`,`html`,`pdf`,`md`等
- 优化了`DenseRetriever`，支持索引构建，增量追加以及索引保存，保存内容包括文档、向量以及索引
- 添加`ReRank`的BGE排序、Rewriter的`HyDE`
- 添加`Judge`的BgeJudge,判断文章是否有用 `20240711`

## 🚀快速上手

### 安装环境

```shell
pip install -r requirements.txt
```

### 1 文档解析

目前支持解析的文件类型包括：`text`,`docx`,`ppt`,`excel`,`html`,`pdf`,`md`

```python
from gomate.modules.document.common_parser import CommonParser

parser = CommonParser()
document_path = 'docs/夏至各地习俗.docx'
chunks = parser.parse(document_path)
print(chunks)
```

### 2 构建检索器

```python
import pandas as pd
from tqdm import tqdm

from gomate.modules.retrieval.dense_retriever import DenseRetriever, DenseRetrieverConfig

retriever_config = DenseRetrieverConfig(
    model_name_or_path="bge-large-zh-v1.5",
    dim=1024,
    index_dir='dense_cache'
)
config_info = retriever_config.log_config()
print(config_info)

retriever = DenseRetriever(config=retriever_config)

data = pd.read_json('docs/zh_refine.json', lines=True)[:5]
print(data)
print(data.columns)

retriever.build_from_texts(documents)
```

保存索引

```python
retriever.save_index()
```

### 3 检索文档

```python
result = retriever.retrieve("RCEP具体包括哪些国家")
print(result)
```

### 4 大模型问答

```python
from gomate.modules.generator.llm import GLMChat

chat = GLMChat(path='THUDM/chatglm3-6b')
print(chat.chat(question, [], content))
```

### 5 添加文档

```python
for documents in tqdm(data['positive'], total=len(data)):
    for document in documents:
        retriever.add_text(document)
for documents in tqdm(data['negative'], total=len(data)):
    for document in documents:
        retriever.add_text(document)
```

## 🔧定制化RAG

> 构建自定义的RAG应用

```python
import os

from gomate.modules.document.common_parser import CommonParser
from gomate.modules.generator.llm import GLMChat
from gomate.modules.reranker.bge_reranker import BgeReranker
from gomate.modules.retrieval.dense_retriever import DenseRetriever


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
# 修改成自己的配置！！！
app_config = ApplicationConfig()
app_config.docs_path = "./docs/"
app_config.llm_model_path = "/data/users/searchgpt/pretrained_models/chatglm3-6b/"

retriever_config = DenseRetrieverConfig(
    model_name_or_path="/data/users/searchgpt/pretrained_models/bge-large-zh-v1.5",
    dim=1024,
    index_dir='/data/users/searchgpt/yq/GoMate/examples/retrievers/dense_cache'
)
rerank_config = BgeRerankerConfig(
    model_name_or_path="/data/users/searchgpt/pretrained_models/bge-reranker-large"
)

app_config.retriever_config = retriever_config
app_config.rerank_config = rerank_config
application = RagApplication(app_config)
application.init_vector_store()
```

```shell
python app.py
```

浏览器访问：[127.0.0.1:7860](127.0.0.1:7860)
![demo.png](resources%2Fdemo.png)

app后台日志：

![app_logging.png](resources%2Fapp_logging.png)

## ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=gomate-community/GoMate&type=Date)](https://star-history.com/#gomate-community/GoMate&Date)

## 研究与开发团队

本项目由网络数据科学与技术重点实验室[`GoMate`](https://github.com/gomate-community)团队完成，团队指导老师为郭嘉丰、范意兴研究员。

## 技术交流群

欢迎多提建议、Bad cases，欢迎进群及时交流，也欢迎大家多提PR</br>

<img src="https://github.com/gomate-community/GoMate/blob/pipeline/resources/wechat.png" width="180px" height="270px">


群满或者合作交流可以联系：

<img src="https://raw.githubusercontent.com/yanqiangmiffy/Chinese-LangChain/master/images/personal.jpg" width="180px">
