import time
from typing import List, Any
import requests
from tqdm import tqdm
from trustrag.modules.rewriter.base import BaseRewriter
import json
import re


class OpenaiRewriterConfig:
    """
    """

    def __init__(self, api_url='http://gomatellm-service.aicloud-yanqiang.svc.cluster.local'):
        self.api_url = api_url

    def log_config(self):
        # Log the current configuration settings
        return f"""
        BgeRewriterConfig:
            API URL: {self.api_url}
        """


class OpenaiRewriter(BaseRewriter):
    """
    A Rewriter that utilizes a BERT-based model for sequence classification
    to judge a list of documents based on their relevance to a given query.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.api_url = self.config.api_url
        print('Successful Init ChatGPT Rewriter ')

    def parse_response(self, response_data: str):
        """
        解析JSON响应字符串，如果解析失败则返回默认空值
        Args:
            json_string (str): JSON格式的响应字符串

        Returns:
            dict: 包含location、date和event字段的字典，解析失败时返回空值
        """
        # 定义默认的返回结构
        default_response = {
            "location": "",
            "date": "",
            "event": ""
        }

        try:
            # 如果response是字符串，则需要再次解析
            if isinstance(response_data, str):
                try:
                    response_data = re.sub(r'^.*?```json\n|```$', '', response_data, flags=re.DOTALL)
                    response_data = json.loads(response_data)
                except json.JSONDecodeError:
                    print("报错")
                    return default_response

            # 从解析后的数据中提取字段，如果不存在则使用空字符串
            result = {
                "location": response_data.get("location", ""),
                "date": response_data.get("date", ""),
                "event": response_data.get("event", "")
            }

            return result

        except json.JSONDecodeError:
            print("报错")
            return default_response
        except Exception:
            print("报错")
            return default_response

    def rewrite(self, query):
        system_prompt = """
请分析用户问题并提取其中的地点、时间、活动或会议名称，将这些信息以JSON格式输出。如果信息不全或用户未提及，则标记为""。按以下格式生成JSON输出：
        {
            "location": "地点或国家名称（如有）",
            "date": "时间日期或者时间范围（比如最新或最近等）（如有）",
            "event": "活动或会议名称（如有）"
        }

        示例：
        输入："在“一带一路”国际合作高峰论坛上，习近平讲了什么？"
        输出：{
            "location": "",
            "date": "",
            "event": "一带一路国际合作高峰论坛"
        }

        输入："在全国卫生与健康大会上，习近平对医疗卫生服务体系改革有哪些具体部署？"
        输出：{
            "location": "",
            "date": "",
            "event": "全国卫生与健康大会"
        }

        输入："在中央外事工作会议上，习对对外工作和外交战略有哪些具体部署？"
        输出：{
            "location": "",
            "date": "",
            "event": "中央外事工作会议"
        }

        输入："习近平在福建考察有什么重要指示？"
        输出：{
            "location": "福建",
            "date": "",
            "event": ""
        }
        
        输入："习近平在福建考察有什么重要指示？"
        输出：{
            "location": "福建",
            "date": "",
            "event": ""
        }
        输入："习近平关于新质生产力有什么最新的论述？"
        输出：{
            "location": "",
            "date": "最新",
            "event": ""
        }
        
        用户问题：
        """

        # Request payload
        payload = {
            "prompt": system_prompt + "\n" + query,
            "teampture": 0.2,
            "top_k": 20
        }
        print(system_prompt + "\n" + query)
        response = requests.post(self.api_url, json=payload)
        response = response.json()
        response = self.parse_response(response['response'])
        return response

