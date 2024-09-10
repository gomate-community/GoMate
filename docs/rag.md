## 基于GoMate的RAG应用构建

### [大模型RAG智能问答挑战赛](https://challenge.xfyun.cn/topic/info?type=RAG-quiz)

> 赛题需要参赛选手设计并实现一个RAG模型，该模型能够从给定的问题出发，检索知识库中的相关信息。利用检索到的信息，结合问题本身，生成准确、全面、权威的回答。

#### 1.数据说明

数据集还可能包括一些未标注的文本，需要参赛者使用RAG技术中的检索增强方法来找到相关信息，并生成答案。这要求参赛者不仅要有强大的检索能力，还要能够生成准确、连贯且符合上下文的文本。

测试集为模拟生成的用户提问，需要参赛选手结合提问和语料完成回答。需注意，在问题中存在部分问题无法回答，需要选手设计合适的策略进行拒绝回答的逻辑。

• corpus.txt.zip：语料库，每行为一篇新闻

• test_question.csv：测试提问

#### 2.评审规则

对于测试提问的回答，采用字符重合比例进行评价，分数最高为1。

#### 3.提交结果

| 序号 | 方法                                             | 分数      |
|----|------------------------------------------------|---------|
| 0  | baseline:glm4_plus                             | 0.34    |
| 1  | bm25s                                          | 0.06091 |
| 2  | bm25s+修改prompt                                 | 0.26175 |
| 3  | bm25s+hybrid检索器+qwen27b                        | 0.22371 |
| 4  | bm25s+hybrid检索器+qwen21.5b                      | 0.23608 |
| 5  | dense+qwen21.5b                                | 0.24613 |
| 6  | hybrid检索器                                      | 0.05696 |
| 7  | hybrid检索器+qwen7b +xunfei prompt                | 0.33623 |
| 8  | hybrid检索器+top10+qwen7b +xunfei prompt          | 0.32735 |
| 9  | hybrid检索器+top5+glm4-9b +glm prompt             | 0.37147 |
| 10 | hybrid检索器+top5+glm4-9b +xunfei prompt          | 0.41775 |
| 11 | hybrid检索器+top5+glm4-9b +xunfei prompt+【无法回答】   | 0.3878  |
| 12 | hybrid检索器+top5+glm4-9b +qwen prompt+rerank :失误 | 0.27884 |
