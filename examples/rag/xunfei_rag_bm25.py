import pickle

import pandas as pd
from tqdm import tqdm

from gomate.modules.document.chunk import TextChunker
from gomate.modules.document.txt_parser import TextParser
from gomate.modules.document.utils import PROJECT_BASE
from gomate.modules.generator.llm import GLM4Chat
from gomate.modules.reranker.bge_reranker import BgeRerankerConfig, BgeReranker
from gomate.modules.retrieval.bm25s_retriever import BM25RetrieverConfig, BM25Retriever


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

    # test_path="H:/2024-Xfyun-RAG/data/test_question.csv"
    # embedding_model_path="H:/pretrained_models/mteb/bge-m3"
    # llm_model_path="H:/pretrained_models/llm/Qwen2-1.5B-Instruct"

    test_path = "/data/users/searchgpt/yq/GoMate_dev/data/competitions/xunfei/test_question.csv"
    embedding_model_path = "/data/users/searchgpt/pretrained_models/bge-large-zh-v1.5"
    llm_model_path = "/data/users/searchgpt/pretrained_models/glm-4-9b-chat"
    # ====================文件解析+切片=========================
    # generate_chunks()
    with open(f'{PROJECT_BASE}/output/chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)
    corpus = []
    for chunk in chunks:
        corpus.extend(chunk)

    # ====================检索器配置=========================
    # BM25 and Dense Retriever configurations
    bm25_config = BM25RetrieverConfig(
        method='lucene',
        index_path='indexs/description_bm25.index',
        k1=1.6,
        b=0.7
    )
    bm25_config.validate()
    print(bm25_config.log_config())

    bm25_retriever = BM25Retriever(bm25_config)
    bm25_retriever.load_index()
    # ====================检索测试=========================
    query = "支付宝"
    results = bm25_retriever.retrieve(query, top_k=10)
    print(len(results))
    # Output results
    for result in results:
        print(f"Text: {result['text']}, Score: {result['score']}")

    # ====================排序配置=========================
    reranker_config = BgeRerankerConfig(
        model_name_or_path="/data/users/searchgpt/pretrained_models/bge-reranker-large"
    )
    bge_reranker = BgeReranker(reranker_config)

    # ====================生成器配置=========================
    # qwen_chat = QwenChat(llm_model_path)
    glm4_chat = GLM4Chat(llm_model_path)

    # ====================检索问答=========================
    test = pd.read_csv(test_path)
    answers = []
    for question in tqdm(test['question'], total=len(test)):
        search_docs = bm25_retriever.retrieve(question, top_k=5)
        search_docs = bge_reranker.rerank(
            query=question,
            documents=[doc['text'] for idx, doc in enumerate(search_docs)]
        )
        # print(search_docs)
        content = '\n'.join([f'信息[{idx}]：' + doc['text'] for idx, doc in enumerate(search_docs)])
        answer = glm4_chat.chat(prompt=question, content=content)
        answers.append(answer[0])
        print(question)
        print(answer[0])
        print("************************************/n")
    test['answer'] = answers

    test[['answer']].to_csv(f'{PROJECT_BASE}/output/gomate_baseline.csv', index=False)