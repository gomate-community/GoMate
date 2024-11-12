import time
from typing import List, Any

import requests
from tqdm import tqdm

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
        判断给定文章是否回答了查询，请严格遵循以下步骤并返回结果 1 或 0：

        1. 会议或活动名称检查：
           - 如果查询包含会议或活动的名称，请检查文章标题是否提到会议相关名称。
             - 若标题提到该名称，返回 1；若未提到该名称，返回 0。
             - 如果查询中出现会议或者活动名称，需要严格遵守上面规则，不确定是否提及名称返回0

        2. 相关性检查：
           - 如果查询中没有会议或活动名称，则判断文章内容是否与查询相关。
             - 若内容相关，返回 1；若不相关，返回 0。

        注意：只返回 1 或 0，不解释原因，不输出其他内容。
        
        示例1：
        查询：在“一带一路”国际合作高峰论坛上，习近平讲了什么？
        新闻：标题：“一带一路”国际合作高峰论坛举行圆桌峰会 习近平主持会议并致辞日期：2017-05-16内容：习近平指出，中方主办这次高峰论坛，目的就是共商合作大计，共建合作平台，共享合作成果，让“一带一路”建设更好造福各国人民。
        返回：1
        
        示例2：
        查询：在“一带一路”国际合作高峰论坛上，习近平讲了什么？
        新闻：习近平：加快推进丝绸之路经济带和21世纪海上丝绸之路建设日期：2014-11-06内容：习近平发表重要讲话强调，丝绸之路经济带和21世纪海上丝绸之路倡议顺应了时代要求和各国加快发展的愿望，提供了一个包容性巨大的发展平台，具有深厚历史渊源和人文基础，能够把快速发展的中国经济同沿线国家的利益结合起来。
        返回：0
        
        示例3：
        查询：在全国卫生与健康大会上，习对医疗卫生服务体系改革有哪些具体部署？
        新闻：标题：习近平在教育文化卫生体育领域专家代表座谈会上的讲话日期：2020-09-23内容：习近平强调，要深化医疗卫生体制改革，加快健全分级诊疗制度、现代医院管理制度、全民医保制度、药品供应保障制度、综合监管制度，合理制定并落实公立医疗卫生机构人员编制标准并建立动态核增机制。
        返回：0
        
        示例3：
        查询：在全国卫生与健康大会上，习对医疗卫生服务体系改革有哪些具体部署？
        新闻：标题：习近平：构建起强大的公共卫生体系 为维护人民健康提供有力保障日期：2020-09-15内容：习近平指出，要改革完善疾病预防控制体系。疾病预防控制体系是保护人民健康、保障公共卫生安全、维护经济社会稳定的重要保障。要立足更精准更有效地防，在理顺体制机制、明确功能定位、提升专业能力等方面加大改革力度。
        返回：0

        查询：{query}
        文章：{passage}
        """

        print("Start Judge query and doc")
        print(query)
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
