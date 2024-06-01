#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: exmaple.py
@time: 2024/05/22
@contact: yanqiangmiffy@gamil.com
"""
from gomate.modules.document.reader import ReadFiles
from gomate.modules.generator.llm import GLMChat
from gomate.modules.retrieval.embedding import BgeEmbedding
from gomate.modules.store import VectorStore

# step1：Document
docs = ReadFiles('./data/docs').get_content(max_token_len=600, cover_content=150)
vector = VectorStore(docs)

# step2：Extract Embedding
embedding = BgeEmbedding("/data/users/searchgpt/pretrained_models/bge-large-zh-v1.5")  # 创建EmbeddingModel
vector.get_vector(EmbeddingModel=embedding)
vector.persist(path='storage')  # 将向量和文档内容保存到storage目录下，下次再用就可以直接加载本地的数据库
vector.load_vector(path='storage')  # 加载本地的数据库

# step3:retrieval
question = '伊朗坠机事故原因是什么？'
contents = vector.query(question, EmbeddingModel=embedding, k=1)
content = '\n'.join(contents[:5])
print(contents)

# step4：QA
chat = GLMChat(path='/data/users/searchgpt/pretrained_models/chatglm3-6b')
print(chat.chat(question, [], content))

# step5 追加文档
docs = ReadFiles('').get_content_by_file(file='data/伊朗问题.txt', max_token_len=600, cover_content=150)
vector.add_documents('storage', docs, embedding)
question = '如今伊朗人的经济生活状况如何？'
contents = vector.query(question, EmbeddingModel=embedding, k=1)
content = '\n'.join(contents[:5])
print(contents)
print(chat.chat(question, [], content))

