import json
import os

import pandas as pd
import requests
from tqdm import tqdm


class LLMCompressApi():
    def __init__(self):
        self.prompt_template = """
        分析以下新闻标题列表,提取它们的共同主题。生成一个简洁、准确且不超过10个字的主题标题。
        新闻标题:
        {titles}
        主题标题:
        """
        # self.api_url = ''
        # 根据自己api地址修改
        self.api_url = 'http://10.208.63.29:8888'

    def compress(self, titles):
        titles = "\n".join(titles)
        prompt = self.prompt_template.format(titles=titles)
        # ====根据自己的api接口传入和输出修改====

        data = {
            "prompt": prompt,
        }
        # loguru.logger.info(data)
        post_json = json.dumps(data)
        response = requests.post(self.api_url, data=post_json, timeout=600)  # v100-2
        response = response.json()
        # =====根据自己的api接口传入和输出修改===

        return response


class LLMReportApi():
    def __init__(self):
        self.prompt_template = """
        请根据以下提供的新闻素材，编写一份主题报告，内容贴切主题内容，不少于50字。
        
        新闻素材:
        {contexts}
        
        主题报告:
        """
        # self.api_url = ''
        # 根据自己api地址修改
        self.api_url = 'http://10.208.63.29:8888'

    def compress(self, titles, contents):
        contexts = ''
        for title, content in zip(titles, contents):
            contexts += f'标题：{title}，"新闻内容：{content}\n'

        prompt = self.prompt_template.format(contexts=contexts)[:4096]
        # ====根据自己的api接口传入和输出修改====

        data = {
            "prompt": prompt,
        }
        # loguru.logger.info(data)
        post_json = json.dumps(data)
        response = requests.post(self.api_url, data=post_json, timeout=600)  # v100-2
        response = response.json()
        # =====根据自己的api接口传入和输出修改===
        return response


dfs = []

for file in os.listdir('level2'):
    if file.endswith('.xlsx'):
        df = pd.read_excel(f'level2/{file}')
        dfs.append(df)

df = pd.concat(dfs, axis=0).reset_index(drop=True)
print(df.columns)
llm_api = LLMCompressApi()
llm_report = LLMReportApi()
with open('result/cluster_level1_index.jsonl', 'w', encoding="utf-8") as f:
    for index, group in tqdm(df.groupby(by=["cluster_level1_index"])):
        titles = group['title'][:30].tolist()
        response1 = llm_api.compress(titles)
        titles = group['title'][:5].tolist()
        contents = group['title'][:5].tolist()
        response2 = llm_report.compress(titles, contents)

        f.write(json.dumps({"cluster_level1_index": index, "level1_title": response1["response"].strip(),
                            "level1_content": response2["response"].strip()}, ensure_ascii=False) + "\n")

with open('result/cluster_level2_index.jsonl', 'w', encoding="utf-8") as f:
    for index, group in tqdm(df.groupby(by=["cluster_level2_index"])):
        titles = group['title'][:30].tolist()
        response1 = llm_api.compress(titles)
        titles = group['title'][:5].tolist()
        contents = group['title'][:5].tolist()
        response2 = llm_report.compress(titles, contents)
        f.write(json.dumps({"cluster_level2_index": index, "level2_title": response1["response"].strip(),
                            "level2_content": response2["response"].strip()}, ensure_ascii=False) + "\n")
