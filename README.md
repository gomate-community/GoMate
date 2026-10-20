# TrustRAG:The RAG Framework within Reliable input,Trusted output
A Configurable and Modular RAG Framework.

\[ English | [中文](README_zh.md) \]


[![Python](https://img.shields.io/badge/Python-3.10.0-3776AB.svg?style=flat)](https://www.python.org)
![workflow status](https://github.com/gomate-community/rageval/actions/workflows/makefile.yml/badge.svg)
[![codecov](https://codecov.io/gh/gomate-community/TrustRAG/graph/badge.svg?token=eG99uSM8mC)](https://codecov.io/gh/gomate-community/TrustRAG)
[![pydocstyle](https://img.shields.io/badge/pydocstyle-enabled-AD4CD3)](http://www.pydocstyle.org/en/stable/)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)


## 🔥Introduction to TrustRAG

TrustRAG is a configurable and modular Retrieval-Augmented Generation (RAG) framework designed to provide **reliable input and trusted output**, ensuring users can obtain high-quality and trustworthy results in retrieval-based question-answering scenarios.

The core design of TrustRAG lies in its **high configurability and modularity**, allowing users to flexibly adjust and optimize each component according to specific needs to meet the requirements of various application scenarios.

## 🔨TrustRAG Framework

![framework.png](resources%2Fframework.png)

## DeepResearch Features

The [Python DeepResearch workflow](api/agent/README.md) supports free web search
through Parallel Search MCP, with no search API key required. Existing provider
selections and configured Serper/Tavily credentials remain in use.

The DeepResearch framework achieves deep information search and processing through layered queries, recursive iteration, and intelligent decision-making. This process includes the following key steps:

1. Intent Understanding
   After the user inputs a query, the system parses it into multiple sub-queries to more precisely understand the user's needs.

2. Processing Condition Judgment
   The system determines whether to continue execution based on the following conditions:
   1. **Whether the token budget is exceeded**
   2. **Whether the action depth is exceeded**
   > If these conditions are met, the query is terminated and the answer is returned directly; otherwise, it enters the recursive execution step.

3. Recursive Execution Steps
   During recursive execution, the system performs information retrieval, model reasoning, and context processing tasks
   **Information Retrieval**
   - **Get current question**
   - **Build question execution sequence**
   - **Recursive traversal**
   - **Depth-first search**
   - **Model reasoning**
     > The system performs model reasoning, judging the next action through system prompts and context understanding.

4. Action Type Determination
   Based on the reasoning results, the system decides the next type of action to execute:
   - **answer**: Answer action
   - **reflect**: Reflection action
   - **search**: Search action
   - **read**: Reading action
   - **coding**: Code action

   > These actions affect the context and continuously update the system state.

5. Result Feedback
   Based on the final action type, the system performs the corresponding task and returns the results to the user, completing the entire process.

DeepResearch process diagram:

![DeepSearch.png](resources/DeepSearch.png)

Run the CLI tool:
```bash
cd trustrag/modules/deepsearch
cp .env.example .env #Configure LLM API and search
python pipeline.py
```

## ✨Key Features

**“Reliable input, Trusted output”**

## 🎉 Update Log
- 📑 **2025.3.8** Supports **Deep Search**, enables slow thinking, and generates research reports.
- 🌐 **2025.3.4** Added `websearch` engine for online searches, supporting **DuckDuck** and **Searxn**
- 🐳 **2025.2.27** Added `Dockerfile`, enabling `Docker` deployment
- 🔍 **2025.2.26** Implemented **large model citation generation**
- 🤖 **2025.2.18** Full integration of `OpenAI` applications, see details in [app.py](app.py)
- 🏆 **2025.1.20** Added support for **vector database engines**, such as `Milvus` and `Qdrant`
- 🖼️ **Multimodal RAG-based Q&A** using **GLM-4V-Flash**, code available at [trustrag/applications/rag_multimodal.py](trustrag/applications/rag_multimodal.py)
- 📦 **TrustRAG packaging and deployment**, supports both `pip` and `source` installations
- 📑 **Added [MinerU Document Parsing](https://github.com/gomate-community/TrustRAG/blob/main/docs/mineru.md)**  
  ➡️ An open-source, high-quality data extraction tool supporting `PDFs`, `web pages`, and `multi-format e-books` **[2024.09.07]**
- 🌲 **Implemented RAPTOR: Recursive Tree Retriever**
- 📂 **Supports modularized parsing of multiple file formats**, including `text`, `docx`, `ppt`, `excel`, `html`, `pdf`, and `md`
- ⚡ **Optimized `DenseRetriever`**, supporting index construction, incremental additions, and index storage, including documents, vectors, and indexes
- 🎯 **Added `ReRank` with `BGE` sorting and `Rewriter` with `HyDE`**
- 🏛️ **Introduced `Judge` module with `BgeJudge`** to assess article relevance **[2024.07.11]**

## 🚀Quick Start

## 🛠️ Installation

### Method 1: Install via `pip`

1. Create a conda environment (optional)

```shell
conda create -n trustrag python=3.12.0
conda activate trustrag
```

2. Install dependencies using `pip`

```shell
pip install trustrag   
```

### Method 2: Install from source

1. Download the source code

```shell
git clone https://github.com/gomate-community/TrustRAG.git
```

2. Install dependencies

```shell
pip install -e . 
```

## 🚀 Quick Start

### 1 Module Overview📝

```text
├── applications
├── modules
|      ├── citation: Answer and evidence citation
|      ├── document: Document parsing and chunking, supports multiple document types
|      ├── generator: Generator
|      ├── judger: Document selection
|      ├── prompt: Prompts
|      ├── refiner: Information summarization
|      ├── reranker: Ranking module
|      ├── retrieval: Retrieval module
|      └── rewriter: Rewriting module
```

### 2 Import Modules

```python
import pickle
import pandas as pd
from tqdm import tqdm

from trustrag.modules.document.chunk import TextChunker
from trustrag.modules.document.txt_parser import TextParser
from trustrag.modules.document.utils import PROJECT_BASE
from trustrag.modules.generator.llm import GLM4Chat
from trustrag.modules.reranker.bge_reranker import BgeRerankerConfig, BgeReranker
from trustrag.modules.retrieval.bm25s_retriever import BM25RetrieverConfig
from trustrag.modules.retrieval.dense_retriever import DenseRetrieverConfig
from trustrag.modules.retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverConfig
```

### 3 Document Parsing and Chunking

```text
def generate_chunks():
    tp = TextParser()  # Represents txt format parsing
    tc = TextChunker()
    paragraphs = tp.parse(r'H:/2024-Xfyun-RAG/data/corpus.txt', encoding="utf-8")
    print(len(paragraphs))
    chunks = []
    for content in tqdm(paragraphs):
        chunk = tc.chunk_sentences([content], chunk_size=1024)
        chunks.append(chunk)

    with open(f'{PROJECT_BASE}/output/chunks.pkl', 'wb') as f:
        pickle.dump(chunks, f)
```
> Each line in `corpus.txt` is a news paragraph. You can customize the logic for reading paragraphs. The corpus is from [Large Model RAG Intelligent Question-Answering Challenge](https://challenge.xfyun.cn/topic/info?type=RAG-quiz&option=zpsm).

`TextChunker` is the text chunking program, primarily using [InfiniFlow/huqie](https://huggingface.co/InfiniFlow/huqie) as the text retrieval tokenizer, suitable for RAG scenarios.

### 4 Building the Retriever

**Configuring the Retriever:**

Below is a reference configuration for a hybrid retriever `HybridRetriever`, where `HybridRetrieverConfig` is composed of `BM25RetrieverConfig` and `DenseRetrieverConfig`.

```python
# BM25 and Dense Retriever configurations
bm25_config = BM25RetrieverConfig(
    method='lucene',
    index_path='indexs/description_bm25.index',
    k1=1.6,
    b=0.7
)
bm25_config.validate()
print(bm25_config.log_config())
dense_config = DenseRetrieverConfig(
    model_name_or_path=embedding_model_path,
    dim=1024,
    index_path='indexs/dense_cache'
)
config_info = dense_config.log_config()
print(config_info)
# Hybrid Retriever configuration
# Since the score frameworks are not on the same dimension, it is recommended to merge them
hybrid_config = HybridRetrieverConfig(
    bm25_config=bm25_config,
    dense_config=dense_config,
    bm25_weight=0.7,  # BM25 retrieval result weight
    dense_weight=0.3  # Dense retrieval result weight
)
hybrid_retriever = HybridRetriever(config=hybrid_config)
```

**Building the Index:**

````python
# Build the index
hybrid_retriever.build_from_texts(corpus)
# Save the index
hybrid_retriever.save_index()
````

If the index is already built, you can skip the above steps and directly load the index:
```text
hybrid_retriever.load_index()
```

**Retrieval Test:**

```python
query = "Alipay"
results = hybrid_retriever.retrieve(query, top_k=10)
print(len(results))
# Output results
for result in results:
    print(f"Text: {result['text']}, Score: {result['score']}")
```

### 5 Ranking Model
<details>
<summary>Bge-Rerank</summary>

We have use [bge-reranker](https://github.com/FlagOpen/FlagEmbedding) as our base reranker model.
```python
from trustrag.modules.reranker.bge_reranker import BgeReranker, BgeRerankerConfig
reranker_config = BgeRerankerConfig(
    model_name_or_path='llms/bge-reranker-large'
)
bge_reranker = BgeReranker(reranker_config)
```
</details>

<details>
<summary>PointWise-Rerank</summary>
We have two pointwise methods so far:

`relevance generation`: LLMs are prompted to judge whether the given query and document are relevant. Candidate documents are reranked based on the likelihood of generating a "yes" response by LLMs. It is the rerank method used in [Holistic Evaluation of Language Models](https://arxiv.org/pdf/2211.09110).

`query generation`: LLMs are prompted to generate a pseudo-query based on the given document. Candidate documents are reranked based on the likelihood of generating the target query by LLMs. It is the rerank method used in [Improving Passage Retrieval with Zero-Shot Question Generation](https://arxiv.org/pdf/2204.07496).

We have implemented [flan-t5](https://huggingface.co/docs/transformers/model_doc/flan-t5) as our pointwise reranker model.
```python
from trustrag.modules.reranker.llm_reranker import LLMRerankerConfig, PointWiseReranker
reranker_config = LLMRerankerConfig(
    model_name_or_path="flan-t5-small"
)
llm_reranker = PointWiseReranker(reranker_config)
```
</details>

<details>
<summary>PairWise-Rerank</summary>
We have two pairwise method so far:

`allpair`: LLMs are prompted to judge which document is the more relevant to the given query than the other. Candidate documents are based on the number of that they win. It is the rerank method used in [Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting](https://arxiv.org/pdf/2306.17563).

`bubblesort`: LLMs are prompted to judge which document is the more relevant to the given query than the other. Candidate documents are reranked using bubblesort algorithm. It is the other rerank method used in [Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting](https://arxiv.org/pdf/2306.17563).

```python
from trustrag.modules.reranker.llm_reranker import LLMRerankerConfig, PairWiseReranker
reranker_config = LLMRerankerConfig(
    model_name_or_path="qwen2-7B-instruct"
)
llm_reranker = PairWiseReranker(reranker_config)
```
</details>

<details>
<summary>ListWise-Rerank</summary>
Waiting to implement...
</details>

<details>
<summary>TourRank</summary>
Waiting to implement...
</details>

<details>
<summary>SetWise-Rerank</summary>
We have one setwise method so far:

`setwise likelihood`: LLMs are prompted to judge which document is the most relevant to the given query. Candidate documents are reranked based on the likelihood of generating the label as the most relevant document by LLMs. It is the base rerank method used in [A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models](https://arxiv.org/pdf/2310.09497).

```python
from trustrag.modules.reranker.llm_reranker import LLMRerankerConfig, SetWiseReranker
reranker_config = LLMRerankerConfig(
    model_name_or_path="qwen2-7B-instruct"
)
llm_reranker = SetWiseReranker(reranker_config)
```
</details>

For more details, please refer to [reranker inference](./examples/rerankers/).

### 6 Generator Configuration
```python
glm4_chat = GLM4Chat(llm_model_path)
```

### 6 Retrieval Question-Answering

```python
# ====================Retrieval Question-Answering=========================
test = pd.read_csv(test_path)
answers = []
for question in tqdm(test['question'], total=len(test)):
    search_docs = hybrid_retriever.retrieve(question, top_k=10)
    search_docs = bge_reranker.rerank(
        query=question,
        documents=[doc['text'] for idx, doc in enumerate(search_docs)]
    )
    # print(search_docs)
    content = '\n'.join([f'Information[{idx}]：' + doc['text'] for idx, doc in enumerate(search_docs)])
    answer = glm4_chat.chat(prompt=question, content=content)
    answers.append(answer[0])
    print(question)
    print(answer[0])
    print("************************************/n")
test['answer'] = answers

test[['answer']].to_csv(f'{PROJECT_BASE}/output/gomate_baseline.csv', index=False)
```

## 🔧Customizing RAG

> Building a custom RAG application

```python
import os

from trustrag.modules.document.common_parser import CommonParser
from trustrag.modules.generator.llm import GLMChat
from trustrag.modules.reranker.bge_reranker import BgeReranker
from trustrag.modules.retrieval.dense_retriever import DenseRetriever


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

The module can be found at [rag.py](trustrag/applications/rag.py)

### 🌐Experience RAG Effects

You can configure the local model path

```text
# Modify to your own configuration!!!
app_config = ApplicationConfig()
app_config.docs_path = "./docs/"
app_config.llm_model_path = "/data/users/searchgpt/pretrained_models/chatglm3-6b/"

retriever_config = DenseRetrieverConfig(
    model_name_or_path="/data/users/searchgpt/pretrained_models/bge-large-zh-v1.5",
    dim=1024,
    index_dir='/data/users/searchgpt/yq/TrustRAG/examples/retrievers/dense_cache'
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

Access via browser: [127.0.0.1:7860](127.0.0.1:7860)

![corpus_demo.png](resources%2Fcorpus_demo.png)
![chat_demo.png](resources%2Fchat_demo.png)

App backend logs:
![app_logging3.png](resources%2Fapp_logging3.png)

## ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=gomate-community/TrustRAG&type=Date)](https://star-history.com/#gomate-community/TrustRAG&Date)

## Research and Development Team

This project is completed by the [`GoMate`](https://github.com/gomate-community) team from the Key Laboratory of Network Data Science and Technology, under the guidance of researchers Jiafeng Guo and Yixing Fan.

## Technical Exchange Group

Welcome to provide suggestions and report bad cases. Join the group for timely communication, and PRs are also welcome.</br>

<img src="https://raw.githubusercontent.com/gomate-community/TrustRAG/pipeline/resources/trustrag_group.png" width="180px">

If the group is full or for cooperation and exchange, please contact:

<img src="https://raw.githubusercontent.com/yanqiangmiffy/Chinese-LangChain/master/images/personal.jpg" width="180px">

## 💗Acknowledgments
>This project thanks the following open-source projects for their support and contributions:
- Document parsing: [infiniflow/ragflow](https://github.com/infiniflow/ragflow/blob/main/deepdoc/README.md)
- PDF file parsing: [opendatalab/MinerU](https://github.com/opendatalab/MinerU)
- Document rerank: [ielab/llm-rankers](https://github.com/ielab/llm-rankers)


## 👉 Citation
```text
@article{fan2025trustrag,
  title={TrustRAG: An Information Assistant with Retrieval Augmented Generation},
  author={Fan, Yixing and Yan, Qiang and Wang, Wenshan and Guo, Jiafeng and Zhang, Ruqing and Cheng, Xueqi},
  journal={arXiv preprint arXiv:2502.13719},
  year={2025},
  url={https://arxiv.org/abs/2502.13719}
}
```
