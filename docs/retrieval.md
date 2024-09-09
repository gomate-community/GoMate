## 检索器

### BM25Retriever

> 基于[`bm25s`](https://github.com/xhluca/bm25s)实现

参数说明:

- `method`：bm25算法：'robertson', 'lucene', 'atire', 'bm25l', 'bm25+'
- `index_path`：向量维度

```python
from gomate.modules.document.common_parser import CommonParser
from gomate.modules.document.utils import PROJECT_BASE
from gomate.modules.retrieval.bm25s_retriever import BM25RetrieverConfig, BM25Retriever

if __name__ == '__main__':

    corpus = []

    new_files = [
        f'{PROJECT_BASE}/data/docs/伊朗.txt',
        f'{PROJECT_BASE}/data/docs/伊朗总统罹难事件.txt',
        f'{PROJECT_BASE}/data/docs/伊朗总统莱希及多位高级官员遇难的直升机事故.txt',
        f'{PROJECT_BASE}/data/docs/伊朗问题.txt',
        f'{PROJECT_BASE}/data/docs/汽车操作手册.pdf',
        # r'H:\2024-Xfyun-RAG\data\corpus.txt'
    ]
    parser = CommonParser()
    for filename in new_files:
        chunks = parser.parse(filename)
        corpus.extend(chunks)

    bm25_config = BM25RetrieverConfig(method='lucene', index_path='indexs/description_bm25.index', k1=1.6, b=0.7)
    bm25_config.validate()
    print(bm25_config.log_config())

    bm25_retriever = BM25Retriever(bm25_config)
    bm25_retriever.build_from_texts(corpus)
    # bm25_retriever.load_index()
    query = "伊朗总统莱希"
    search_docs = bm25_retriever.retrieve(query)
    print(search_docs)
```

### DenseRetriever

参数说明:

- `model_name_or_path`：embedding模型hf名称或者路径
- `dim`：向量维度
- `index_dir`:构建索引路径

```python
import pandas as pd
from tqdm import tqdm

from gomate.modules.retrieval.dense_retriever import DenseRetriever, DenseRetrieverConfig

if __name__ == '__main__':
    retriever_config = DenseRetrieverConfig(
        model_name_or_path="/data/users/searchgpt/pretrained_models/bge-large-zh-v1.5",
        dim=1024,
        index_dir='/data/users/searchgpt/yq/GoMate/examples/retrievers/dense_cache'
    )
    config_info = retriever_config.log_config()
    print(config_info)
    retriever = DenseRetriever(config=retriever_config)
    data = pd.read_json('/data/users/searchgpt/yq/GoMate/data/docs/zh_refine.json', lines=True)[:5]
    print(data)
    print(data.columns)

    corpus = []
    for documents in tqdm(data['positive'], total=len(data)):
        for document in documents:
            # retriever.add_text(document)
            corpus.append(document)
    for documents in tqdm(data['negative'], total=len(data)):
        for document in documents:
            #     retriever.add_text(document)
            corpus.append(document)
    print("len(corpus)", len(corpus))
    retriever.build_from_texts(corpus)
    result = retriever.retrieve("RCEP具体包括哪些国家")
    print(result)
    retriever.save_index()
```

### HybridRetriever

> 混合检索，将Bm25检索以及Dense检索的结果进行合并

```python
from gomate.modules.document.common_parser import CommonParser
from gomate.modules.retrieval.bm25s_retriever import BM25RetrieverConfig
from gomate.modules.retrieval.dense_retriever import DenseRetrieverConfig
from gomate.modules.retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverConfig

if __name__ == '__main__':
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
        model_name_or_path="/data/users/searchgpt/pretrained_models/bge-large-zh-v1.5",
        dim=1024,
        index_path='/data/users/searchgpt/yq/GoMate/examples/retrievers/dense_cache'
    )
    config_info = dense_config.log_config()
    print(config_info)

    # Hybrid Retriever configuration
    hybrid_config = HybridRetrieverConfig(
        bm25_config=bm25_config,
        dense_config=dense_config,
        bm25_weight=0.5,
        dense_weight=0.5
    )
    hybrid_retriever = HybridRetriever(config=hybrid_config)

    # Corpus
    corpus = []

    # Files to be parsed
    new_files = [
        r'/data/users/searchgpt/yq/GoMate_dev/data/docs/伊朗.txt',
        r'/data/users/searchgpt/yq/GoMate_dev/data/docs/伊朗总统罹难事件.txt',
        r'/data/users/searchgpt/yq/GoMate_dev/data/docs/伊朗总统莱希及多位高级官员遇难的直升机事故.txt',
        r'/data/users/searchgpt/yq/GoMate_dev/data/docs/伊朗问题.txt',
        r'/data/users/searchgpt/yq/GoMate_dev/data/docs/新冠肺炎疫情.pdf',
    ]

    # Parsing documents
    parser = CommonParser()
    for filename in new_files:
        chunks = parser.parse(filename)
        corpus.extend(chunks)

    # Build hybrid retriever from texts
    hybrid_retriever.build_from_texts(corpus)

    # Query
    query = "新冠肺炎疫情"
    results = hybrid_retriever.retrieve(query, top_k=3)

    # Output results
    for result in results:
        print(f"Text: {result['text']}, Score: {result['score']}")

```