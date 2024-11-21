#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: mongo_utils.py
@time: 2022/11/18
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import sys
import json
from datetime import date, datetime

import numpy as np
import pymongo
from bson import ObjectId

class MongoConfig:
    client = 'mongodb://root:root@127.0.0.1:6080/'

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
        # client = pymongo.MongoClient("mongodb://root:golaxyintelligence@10.208.61.115:20000/")
        client = pymongo.MongoClient(MongoConfig.client)
        # print(client.database_names())
        db = client['RAG']
        return db

    def get_prompts(self):
        """
        获取任务标签
        """
        collection = self.db['prompts']
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
        result={'id':result.inserted_id}
        result=json.loads(json.dumps(result, cls=NpEncoder, ensure_ascii=False))
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


if __name__ == '__main__':
    mc = MongoCursor()
    print(mc.get_prompts())
    print(mc.find_one("650d1d8d3731e344eca11d63", "docs"))
