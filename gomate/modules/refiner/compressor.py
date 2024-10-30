import json

import loguru
import requests


class LLMCompressApi():
    def __init__(self):
        self.prompt_template = """你是一个文档内容总结专家。你的任务是根据用户提供的查询语句，从给定的文档内容中提取出相关信息，并进行全面的总结，确保覆盖所有重要方面，且与用户的问题密切相关。请按照以下步骤进行：
    1. 阅读并理解用户提供的查询语句，分析其意图和关键内容。
    2. 评估给定的文档内容是否能够准确的回答问题，是否与问题相关：
       - 如果文档内容包含与查询相关的准确信息，继续下一步
       - 如果文档内容与查询不能够准确回答问题中的信息，不要胡编乱造，请直接返回："参考信息没有相关答案内容，无法总结和用户问题相关的内容"
    3. 严格的筛选标准
    - 只有在文档内容能够100%精确匹配查询要素时才进行回答
    - 对于任何模糊、相似或推测的内容一律不采用
    - 特别注意：
     * 会议届次必须完全匹配（如二十届必须是二十届，不能用十九届等其他届次的内容）
     * 时间节点必须精确对应
     * 数据必须在文档中明确存在
     * 人物言论必须有明确出处
    4. 综合提取的信息，生成一个简洁且全面的总结，确保：
       - 直接回答用户的查询要点
       - 按重要性组织信息
       - 保持逻辑清晰
       - 语言简洁准确
    
    ### 输入
    - 查询语句：{query}
    - 文档内容列表：{contexts}
    
    ### 输出
    如果能够回答：
    [总结内容，分段组织]
    
    如果无法回答：
    很抱歉，根据参考信息无法总结和用户问题相关的内容
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
