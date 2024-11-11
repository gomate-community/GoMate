import time
from typing import List, Any
import requests
from tqdm import  tqdm
from gomate.modules.judger.base import BaseJudger

class OpenaiJudgerConfig:
    """
    """
    def __init__(self, api_url='http://gomatellm-service.aicloud-yanqiang.svc.cluster.local'):
        self.api_url = api_url

    def log_config(self):
        # Log the current configuration settings
        return f"""
        BgejudgerConfig:
            API URL: {self.api_url}
        """


class OpenaiJudger(BaseJudger):
    """
    A judger that utilizes a BERT-based model for sequence classification
    to judge a list of documents based on their relevance to a given query.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.api_url = self.config.api_url
        print('Successful Init ChatGPT Judger ')

    def judge(self, query: str, documents: List[str], k: int = 5, is_sorted: bool = False) -> list[dict[str, Any]]:
        system_prompt = """
            给定一个查询 A 和一段文章,判断该文章是否包含对该查询的答案,并提供 1 或 0的预测结果。
            关键步骤如下:

            检查查询是否包含特定会议或活动的名称。
            如果包含,检查文章标题或内容是否提到了相同的会议或活动名称。

            如果提到了会议/活动名称,返回1， 如果没有提到会议/活动名称,返回0。
            如果查询中没有包含特定的会议/活动名称,检查文章内容是否与查询相关。如果内容相关,返回 1，如果内容不相关,返回0。

            只返回预测结果1或者0，不要解释原因，不要输出额外内容

            查询：{query}
            文章：{passage}

            """
        print("Start Judge query and doc")
        start_time = time.time()
        judge_docs = []
        for passage in tqdm(documents):
            # API endpoint
            # Request payload
            payload = {
                "prompt": system_prompt.format(query=query, passage=passage),
                "teampture": 0.1,
                "top_k": 8
            }
            try:
                # Send POST request
                response = requests.post(self.api_url, json=payload)
                # Check if request was successful
                response.raise_for_status()

                judge_docs.append(
                    {
                        'doc': passage,
                        'score': int(response.json()['response'].strip()),
                        'label': int(response.json()['response'].strip())

                    }
                )
            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}")
                judge_docs.append(
                    {
                        'doc': passage,
                        'score': 0,
                        'label': 0
                    }
                )
        end_time = time.time()
        print(end_time - start_time)
        return judge_docs
