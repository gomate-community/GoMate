import pickle

import pandas as pd
from tqdm import tqdm

from gomate.modules.document.chunk import TextChunker
from gomate.modules.document.txt_parser import TextParser
from gomate.modules.document.utils import PROJECT_BASE
from gomate.modules.generator.llm import QwenChat
from gomate.modules.retrieval.bm25s_retriever import BM25RetrieverConfig
from gomate.modules.retrieval.dense_retriever import DenseRetrieverConfig
from gomate.modules.retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverConfig


def generate_chunks():
    tp = TextParser()
    tc = TextChunker()
    paragraphs = tp.parse(r'H:/2024-Xfyun-RAG/data/corpus.txt', encoding="utf-8")
    print(len(paragraphs))
    chunks = []
    for content in tqdm(paragraphs):
        chunk = tc.chunk_sentences([content], chunk_size=1024)
        chunks.append(chunk)

    with open(f'{PROJECT_BASE}/output/chunks.pkl', 'wb') as f:
        pickle.dump(chunks, f)


if __name__ == '__main__':

    with open(f'{PROJECT_BASE}/output/chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)
    corpus = []
    for chunk in chunks:
        corpus.extend(chunk)
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
        model_name_or_path="H:/pretrained_models/mteb/bge-m3",
        dim=1024,
        index_path='indexs/dense_cache'
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
    hybrid_retriever.build_from_texts(corpus)

    # Query
    query = "新冠肺炎疫情"
    results = hybrid_retriever.retrieve(query, top_k=3)

    # Output results
    for result in results:
        print(f"Text: {result['text']}, Score: {result['score']}")

        # 对话
        qwen_chat = QwenChat('H:/pretrained_models/llm/Qwen2-1.5B-Instruct')

    test = pd.read_csv(r'H:/2024-Xfyun-RAG/data/test_question.csv')
    answers = []
    for question in tqdm(test['question'], total=len(test)):
        search_docs = hybrid_retriever.retrieve(question)
        content = '/n'.join([f'信息[{idx}]：' + doc['text'] for idx, doc in enumerate(search_docs)])
        answer = qwen_chat.chat(prompt=question, content=content)
        answers.append(answer[0])
        print(answer[0])
        print("************************************/n")
    test['answer'] = answers

    test[['answer']].to_csv(r'H:/2024-Xfyun-RAG/bm25_v2.csv', index=False)
