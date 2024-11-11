import json

from gomate.modules.judger.chatgpt_judger import OpenaiJudger, OpenaiJudgerConfig

if __name__ == '__main__':

    with open('citation.json', 'r', encoding="utf-8") as f:
        data = json.load(f)

    judger_config = OpenaiJudgerConfig(
        api_url="https://aicloud.oneainexus.cn:30013/inference/aicloud-yanqiang/gomatellm/"
    )
    openai_judger = OpenaiJudger(judger_config)

    documents = [
        f"标题：{doc['newsinfo']['title']}\n日期：{doc['newsinfo']['date']}\n内容：{doc['newsinfo']['content']}\n" for doc
        in data['selected_docs']
    ]
    top_docs = openai_judger.judge(
        query="在“一带一路”国际合作高峰论坛上，习近平讲了什么？",
        documents=documents,
    )
    print(top_docs)
