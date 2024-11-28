import pickle

import pandas as pd
from tqdm import tqdm

from trustrag.modules.document.chunk import TextChunker
from trustrag.modules.document.txt_parser import TextParser
from trustrag.modules.document.utils import PROJECT_BASE
from trustrag.modules.generator.llm import QwenChat
from trustrag.modules.retrieval.dense_retriever import DenseRetrieverConfig,DenseRetriever


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

    test_path="H:/2024-Xfyun-RAG/data/test_question.csv"
    embedding_model_path="H:/pretrained_models/mteb/bge-m3"
    llm_model_path="H:/pretrained_models/llm/Qwen2-1.5B-Instruct"

    # test_path = "/data/users/searchgpt/yq/GoMate_dev/data/competitions/xunfei/test_question.csv"
    # embedding_model_path = "/data/users/searchgpt/pretrained_models/bge-large-zh-v1.5"
    # llm_model_path = "/data/users/searchgpt/pretrained_models/Qwen2-7B-Instruct"

    with open(f'{PROJECT_BASE}/output/chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)
    corpus = []
    for chunk in chunks:
        corpus.extend(chunk)


    dense_config = DenseRetrieverConfig(
        model_name_or_path=embedding_model_path,
        dim=1024,
        index_path='indexs/dense_cache'
    )
    config_info = dense_config.log_config()
    print(config_info)

    dense_retriever = DenseRetriever(dense_config  )
    dense_retriever.load_index()
    # Query
    query = "新冠肺炎疫情"
    results = dense_retriever.retrieve(query, top_k=3)

    # Output results
    for result in results:
        print(f"Text: {result['text']}, Score: {result['score']}")

        # 对话
    qwen_chat = QwenChat(llm_model_path)

    test = pd.read_csv(test_path)
    answers = []
    for question in tqdm(test['question'], total=len(test)):
        search_docs = dense_retriever.retrieve(question)
        content = '/n'.join([f'信息[{idx}]：' + doc['text'] for idx, doc in enumerate(search_docs)])
        answer = qwen_chat.chat(prompt=question, content=content)
        answers.append(answer[0])
        print(answer[0])
        print("************************************/n")
    test['answer'] = answers

    test[['answer']].to_csv(f'{PROJECT_BASE}/output/bm25_v2.csv', index=False)
