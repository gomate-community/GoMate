import os

import pandas as pd
from tqdm import tqdm

from gomate.modules.document.common_parser import CommonParser
from gomate.modules.generator.llm import GLMChat
from gomate.modules.retrieval.dense_retriever import DenseRetriever, DenseRetrieverConfig
from gomate.modules.rewriter.hyde_rewriter import HydeRewriter
from gomate.modules.rewriter.promptor import Promptor

if __name__ == '__main__':
    promptor = Promptor(task="WEB_SEARCH", language="zh")

    retriever_config = DenseRetrieverConfig(
        model_name_or_path="/data/users/searchgpt/pretrained_models/bge-large-zh-v1.5",
        dim=1024,
        index_dir='/data/users/searchgpt/yq/GoMate/examples/retrievers/dense_cache'
    )
    config_info = retriever_config.log_config()
    retriever = DenseRetriever(config=retriever_config)
    parser = CommonParser()

    chunks = []
    docs_path = '/data/users/searchgpt/yq/GoMate_dev/data/docs'
    for filename in os.listdir(docs_path):
        file_path = os.path.join(docs_path, filename)
        try:
            chunks.extend(parser.parse(file_path))
        except:
            pass
    retriever.build_from_texts(chunks)

    data = pd.read_json('/data/users/searchgpt/yq/GoMate/data/docs/zh_refine.json', lines=True)[:5]
    for documents in tqdm(data['positive'], total=len(data)):
        for document in documents:
            retriever.add_text(document)
    for documents in tqdm(data['negative'], total=len(data)):
        for document in documents:
            retriever.add_text(document)

    print("init_vector_store done! ")
    generator = GLMChat("/data/users/searchgpt/pretrained_models/glm-4-9b-chat")

    hyde = HydeRewriter(promptor, generator, retriever)
    hypothesis_document = hyde.rewrite("RCEP具体包括哪些国家")
    print("==================hypothesis_document=================\n")
    print(hypothesis_document)
    hyde_result = hyde.retrieve("RCEP具体包括哪些国家")
    print("==================hyde_result=================\n")
    print(hyde_result['retrieve_result'])
    dense_result = retriever.retrieve("RCEP具体包括哪些国家")
    print("==================dense_result=================\n")
    print(dense_result)
    hyde_answer, _ = generator.chat(prompt="RCEP具体包括哪些国家",
                                    content='\n'.join([doc['text'] for doc in hyde_result['retrieve_result']]))
    print("==================hyde_answer=================\n")
    print(hyde_answer)
    dense_answer, _ = generator.chat(prompt="RCEP具体包括哪些国家",
                                     content='\n'.join([doc['text'] for doc in dense_result]))
    print("==================dense_answer=================\n")
    print(dense_answer)

    print("****" * 20)

    hypothesis_document = hyde.rewrite("数据集类型有哪些？")
    print("==================hypothesis_document=================\n")
    print(hypothesis_document)
    hyde_result = hyde.retrieve("数据集类型有哪些？")
    print("==================hyde_result=================\n")
    print(hyde_result['retrieve_result'])
    dense_result = retriever.retrieve("数据集类型有哪些？")
    print("==================dense_result=================\n")
    print(dense_result)
    hyde_answer, _ = generator.chat(prompt="数据集类型有哪些？",
                                    content='\n'.join([doc['text'] for doc in hyde_result['retrieve_result']]))
    print("==================hyde_answer=================\n")
    print(hyde_answer)
    dense_answer, _ = generator.chat(prompt="数据集类型有哪些？",
                                     content='\n'.join([doc['text'] for doc in dense_result]))
    print("==================dense_answer=================\n")
    print(dense_answer)

    print("****" * 20)

    hypothesis_document = hyde.rewrite("Sklearn可以使用的数据集有哪些？")
    print("==================hypothesis_document=================\n")
    print(hypothesis_document)
    hyde_result = hyde.retrieve("Sklearn可以使用的数据集有哪些？")
    print("==================hyde_result=================\n")
    print(hyde_result['retrieve_result'])
    dense_result = retriever.retrieve("Sklearn可以使用的数据集有哪些？")
    print("==================dense_result=================\n")
    print(dense_result)
    hyde_answer, _ = generator.chat(prompt="Sklearn可以使用的数据集有哪些？",
                                    content='\n'.join([doc['text'] for doc in hyde_result['retrieve_result']]))
    print("==================hyde_answer=================\n")
    print(hyde_answer)
    dense_answer, _ = generator.chat(prompt="Sklearn可以使用的数据集有哪些？",
                                     content='\n'.join([doc['text'] for doc in dense_result]))
    print("==================dense_answer=================\n")
    print(dense_answer)
