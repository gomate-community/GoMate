import sys
sys.path.append("/data/users/searchgpt/yq/GoMate_dev")
import json
import os
import time
from datetime import date
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import numpy as np
import pandas as pd
import pymongo
import requests
from bson import ObjectId
from tqdm import tqdm
import loguru
from singlepass import SGCluster

keywords = [
    "美国",
    "中美贸易",
    "俄乌冲突",
    "中东",
]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, ObjectId):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, date):
            return obj.strftime('%Y-%m-%d')
        elif isinstance(obj, ObjectId):
            return str(obj)
        else:
            return super(NpEncoder, self).default(obj)


class MongoCursor(object):
    def __init__(self):
        self.db = self.get_conn()

    def get_conn(self):
        client = pymongo.MongoClient("mongodb://root:golaxyintelligence@10.208.61.115:20000/")
        # client = pymongo.MongoClient("mongodb://root:golaxyintelligence@10.60.1.145:20000/")
        db = client['goinv3_2409']
        return db

    def get_reports(self):
        """
        获取任务标签
        """
        collection = self.db['report']
        data = [json.loads(json.dumps(task, cls=NpEncoder, ensure_ascii=False)) for task in collection.find()][0]
        # print(data)
        print(data['_id'])
        del data['_id']
        return data

    def insert_one(self, document, collection_name):
        # print(document)
        collection = self.db[collection_name]
        result = collection.insert_one(document)
        # print(result.inserted_id)
        result = {'id': result.inserted_id}
        result = json.loads(json.dumps(result, cls=NpEncoder, ensure_ascii=False))
        return result

    def find_one(self, id, collection_name):
        collection = self.db[collection_name]
        result = collection.find_one({'_id': ObjectId(id)})
        # print(type(result))
        result = json.loads(json.dumps(result, cls=NpEncoder, ensure_ascii=False))
        result = self.rename(result, '_id', 'id')
        return result

    def delete_one(self, id, collection_name):
        collection = self.db[collection_name]
        result = collection.delete_one({'_id': ObjectId(id)})
        return result

    def find_many(self, query, collection_name):
        collection = self.db[collection_name]
        results = collection.find(query, sort=[("update_time", pymongo.DESCENDING)])
        results = [json.loads(json.dumps(task, cls=NpEncoder, ensure_ascii=False)) for task in results]
        results = [self.rename(result, 'title', 'name') for result in results]
        results = [self.rename(result, 'update_time', 'update_date') for result in results]
        return results

    def update_one(self, id, collection_name, **kwargs):
        """

        :param id:
        :param collection_name:
        :param update: {'$set': kwargs}
        :return:
        """
        collection = self.db[collection_name]
        result = collection.update_one(filter={'_id': ObjectId(id)}, update={'$set': kwargs})
        # print(type(result.raw_result))
        message = result.raw_result
        if message['updatedExisting']:
            message['message'] = "保存成功"
        else:
            message['message'] = "保存失败，请检查id是否存在"
        return message

    def rename(self, old_dict, old_name, new_name):
        new_dict = {}
        for key, value in zip(old_dict.keys(), old_dict.values()):
            new_key = key if key != old_name else new_name
            new_dict[new_key] = old_dict[key]
        return new_dict


class LLMCompressApi():
    def __init__(self, type="title"):
        self.type = type
        if self.type == "title":
            self.prompt_template = """
            分析以下新闻标题列表,提取它们的共同主题。生成一个简洁、准确且不超过10个字的主题标题。
            注意：
            1. 生成标题首尾不要带有引号，中间可以带有引号
            2. 如果输入标题内容是英文，请用中文编写
            新闻标题:
            {titles}
            主题标题:
            """
        else:
            self.prompt_template = """
                    请根据以下提供的新闻素材，编写一份主题报告，内容贴切主题内容，如果输入标题内容是英文，请用中文编写，不少于50字。
    
                    新闻素材:
                    {contexts}
    
                    主题报告:
                    """
        self.api_url = "http://10.208.63.29:8888"

    def compress(self, titles, contents):
        if self.type == 'title':
            titles = "\n".join(titles)
            prompt = self.prompt_template.format(titles=titles)
        else:
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


def get_es_data():
    os.makedirs("data/", exist_ok=True)

    for word in keywords:
        loguru.logger.info("正在获取es数据："+word)
        url = f"http://10.208.61.117:9200/document_share_data_30_news/_search?q={word}&size=6000&sort=publish_time:desc"
        response = requests.get(url)
        with open(f"data/{word}_data.json", "w", encoding="utf-8") as f:
            json.dump(response.json(), f, ensure_ascii=False, indent=4)

        with open(f"data/{word}_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        sources = [hit["_source"] for hit in data["hits"]["hits"]]

        source_df = pd.DataFrame(sources)

        source_df["id"] = source_df.index
        source_df["id"] = "source_" + source_df["id"].astype(str)

        source_df[["id", "title", "content"]].to_excel(f"data/{word}_data.xlsx")


def run_cluster_data():
    print("=========一级聚类==========")
    for keyword in keywords:
        loguru.logger.info("一级聚类："+keyword)
        data = pd.read_excel(f"data/{keyword}_data.xlsx", dtype={"id": str})
        data = data.drop_duplicates(subset=["title"]).reset_index(drop=True)
        data["id"] = data["id"].astype(str)
        if not os.path.exists(f"result/level1_{keyword}_result.xlsx"):
            sc = SGCluster(
                vector_path=f"result/level1_{keyword}_vector.npy",
                result_txt_file=f"result/level1_{keyword}_result.txt",
                output_file=f"result/level1_{keyword}_result.xlsx",
                threshold=0.4,
                max_features=8888,
                n_components=1024,
                ngrams=2,
                level=1
            )
            sc.classify(data)
    print("=========二级聚类==========")
    for keyword in keywords:
        loguru.logger.info("二级聚类："+keyword)
        data = pd.read_excel(f"result/level1_{keyword}_result.xlsx", dtype={"id": str})
        data = data.drop_duplicates(subset=["title"]).reset_index(drop=True)
        data["id"] = data["id"].astype(str)
        for cluster_index, group in data.groupby(by="cluster_index"):
            try:
                if len(group) > 4:
                    group = group.reset_index(drop=True)
                    sc = SGCluster(
                        vector_path=f"result/level2_{keyword}_vector_{cluster_index}.npy",
                        result_txt_file=f"result/level2_{keyword}_result_{cluster_index}.txt",
                        output_file=f"result/level2_{keyword}_result_{cluster_index}.xlsx",
                        threshold=0.5,
                        max_features=8888,
                        n_components=64,
                        ngrams=2,
                        level=2
                    )
                    sc.classify(group)
            except:
                pass

def generate_report():
    for keyword in keywords:
        loguru.logger.info("正在生成报告:"+keyword)
        dfs = []
        for file in os.listdir("result"):
            if file.endswith(".xlsx") and keyword in file and 'level2_' in file:
                df = pd.read_excel(f"result/{file}")
                dfs.append(df)
        df = pd.concat(dfs, axis=0).reset_index(drop=True)

        df.to_excel(f"result/{keyword}_cluster_double.xlsx", index=False)
        llm_api = LLMCompressApi(type="title")
        llm_report = LLMCompressApi(type="report")
        if not os.path.exists(f"result/{keyword}_cluster_level1_index.jsonl"):
            with open(f"result/{keyword}_cluster_level1_index.jsonl", "w", encoding="utf-8") as f:
                for index, group in tqdm(df.groupby(by=["cluster_level1_index"])):
                    if len(group)>=3:
                        titles = group["title"][:30].tolist()
                        contents = group["title"][:5].tolist()
                        response1 = llm_api.compress(titles, contents)
                        titles = group["title"][:5].tolist()
                        response2 = llm_report.compress(titles, contents)

                        f.write(json.dumps({"cluster_level1_index": index, "level1_title": response1["response"].strip(),
                                            "level1_content": response2["response"].strip()}, ensure_ascii=False) + "\n")

            with open(f"result/{keyword}_cluster_level2_index.jsonl", "w", encoding="utf-8") as f:
                for index, group in tqdm(df.groupby(by=["cluster_level2_index"])):
                    if len(group)>=3:
                        titles = group["title"][:30].tolist()
                        contents = group["title"][:5].tolist()
                        response1 = llm_api.compress(titles, contents)
                        titles = group["title"][:5].tolist()
                        response2 = llm_report.compress(titles, contents)
                        f.write(json.dumps({"cluster_level2_index": index, "level2_title": response1["response"].strip(),
                                            "level2_content": response2["response"].strip()}, ensure_ascii=False) + "\n")


def insert_mongo_report():
    mc = MongoCursor()
    for idx,keyword in enumerate(keywords):
        try:
            loguru.logger.info("正在插入MongoDB成功："+keyword)
            df = pd.read_excel(f"result/{keyword}_cluster_double.xlsx")
            level1_mapping = {}
            with open(f"result/{keyword}_cluster_level1_index.jsonl", 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    data = json.loads(line.strip())
                    level1_mapping[data['cluster_level1_index']] = {
                        'level1_title': data['level1_title'],
                        'level1_content': data['level1_content'],
                    }

            level2_mapping = {}
            with open(f"result/{keyword}_cluster_level2_index.jsonl", 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    data = json.loads(line.strip())
                    level2_mapping[data['cluster_level2_index']] = {
                        'level2_title': data['level2_title'],
                        'level2_content': data['level2_content'],
                    }

            df['level1_title'] = df['cluster_level1_index'].apply(
                lambda x: level1_mapping.get(x, {}).get('level1_title', ''))
            df['level1_content'] = df['cluster_level1_index'].apply(
                lambda x: level1_mapping.get(x, {}).get('level1_content', ''))
            df['level2_title'] = df['cluster_level2_index'].apply(
                lambda x: level2_mapping.get(x, {}).get('level2_title', ''))
            df['level2_content'] = df['cluster_level2_index'].apply(
                lambda x: level2_mapping.get(x, {}).get('level2_content', ''))

            # 查看结果
            # 获取当前日期并格式化为 YYYYMMDD 格式
            current_date = datetime.now().strftime("%Y%m%d")
            template = {
                '_id': f'{current_date}_00{idx+1}',
                'name': f'开源情报每日简报-{current_date}',
                'description': '',
                'tags': ['开源', '新闻', keyword],
                'content': [],
                'version': '1.0',
                'comment': '',
                'source': 'admin',
                'owner': 'system',
                'created_time': int(time.time() * 1000),
                'modified_time': int(time.time() * 1000)
            }
            contents = []
            for level1_index, group1 in df.groupby(by=["cluster_level1_index"]):
                nodes = []
                for level2_index, group2 in group1.groupby(by=["cluster_level2_index"]):
                    nodes.append(
                        {
                            'title': group2['level2_title'].unique()[0],
                            'content': group2['level2_content'].unique()[0]
                        }
                    )
                contents.append({
                    'title': group1['level1_title'].unique()[0],
                    'content': group1['level1_content'].unique()[0],
                    'nodes': nodes
                })
            template['content'] = contents
            mc.insert_one(template, 'report')
        except:
            loguru.logger.error("插入MongoDB失败:"+keyword)


def run():
    try:
        loguru.logger.info("开始执行任务")
        get_es_data()
        run_cluster_data()
        generate_report()
        insert_mongo_report()
        loguru.logger.info("任务执行完成")
    except Exception as e:
        loguru.logger.error(f"任务执行出错: {str(e)}")


def main():
    scheduler = BlockingScheduler()

    # 设置每天06:00执行任务
    trigger = CronTrigger(
        hour=6,
        minute=0
    )

    scheduler.add_job(
        run,
        trigger=trigger,
        id='daily_job',
        name='每日数据处理任务',
        misfire_grace_time=3600  # 错过执行时间1小时内仍会执行
    )

    loguru.logger.info("调度器已启动，等待执行...")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        loguru.logger.info("调度器已关闭")

def sing_run():
    generate_report()
    insert_mongo_report()
if __name__ == '__main__':
    main()
