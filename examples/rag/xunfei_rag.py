import pickle

import pandas as pd
from tqdm import tqdm

from gomate.modules.document.chunk import TextChunker
from gomate.modules.document.txt_parser import TextParser
from gomate.modules.document.utils import PROJECT_BASE
from gomate.modules.generator.llm import QwenChat
from gomate.modules.retrieval.bm25s_retriever import BM25Retriever
from gomate.modules.retrieval.dense_retriever import DenseRetriever,DenseRetrieverConfig

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

    # 检索器构建
    bm25_retriever = BM25Retriever(method="lucene",
                                   index_path="indexs/description_bm25.index",
                                   rebuild_index=False,
                                   corpus=corpus)

    retriever_config = DenseRetrieverConfig(
        model_name_or_path="H:/pretrained_models/mteb/bge-large-zh-v1.5",
        dim=1024,
        index_dir='H:/Projects/GoMate/examples/rag/indexs/dense_cache'
    )
    config_info = retriever_config.log_config()
    print(config_info)

    retriever = DenseRetriever(config=retriever_config)
    query = "拉莫斯在比赛中表现不佳的原因是什么？"
    search_docs = bm25_retriever.retrieve(query)
    print(search_docs)

    # 对话
    qwen_chat = QwenChat('H:/pretrained_models/llm/Qwen2-1.5B-Instruct')

    test = pd.read_csv(r'H:/2024-Xfyun-RAG/data/test_question.csv')
    answers = []
    for question in tqdm( test['question'],total=len(test)):
        search_docs = bm25_retriever.retrieve(question)
        content = '/n'.join([f'信息[{idx}]：'+doc['text'] for idx,doc in enumerate(search_docs)])
        answer = qwen_chat.chat(prompt=question, content=content)
        answers.append(answer[0])
        print(answer[0])
        print("************************************/n")
    test['answer'] = answers

    test[['answer']].to_csv(r'H:/2024-Xfyun-RAG/bm25_v2.csv', index=False)
