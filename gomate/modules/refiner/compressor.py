import json

import loguru
import requests


class LLMCompressApi():
    def __init__(self):
        self.prompt_template = """
        你是一个文档内容总结专家。你的任务是根据用户提供的查询语句，从给定的文档内容中提取出相关信息，并进行全面的总结，确保覆盖所有重要方面，且与用户的问题密切相关。请按照以下步骤进行：
    
        1. 阅读并理解用户提供的查询语句，分析其意图和关键内容。
        2. 从给定的文档内容中提取与查询最相关的信息。
        3. 综合提取的信息，生成一个简洁且全面的总结，确保回答用户的查询。
    
        ### 输入
        - 查询语句：{query}
        - 文档内容列表：{contexts}
    
        ### 输出:
        """
        # self.api_url = ''
        # 根据自己api地址修改
        self.api_url = 'http://10.208.63.29:8888'

    def compress(self, query, contexts):
        prompt = self.prompt_template.format(query=query, contexts=contexts)
        # ====根据自己的api接口传入和输出修改====

        data = {
            "prompt": prompt,
        }
        loguru.logger.info(data)
        post_json = json.dumps(data)
        response = requests.post(self.api_url, data=post_json, timeout=600)  # v100-2
        response = response.json()
        # =====根据自己的api接口传入和输出修改===

        return response
