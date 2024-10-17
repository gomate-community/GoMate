#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:quincy qiang
# email:yanqiangmiffy@gmail.com
# datetime:2021/5/25 12:59
# description:"do something"
import gc
import logging
import os
import random

import numpy as np
from joblib import Parallel, delayed
from libraries.timer import *
from tqdm import tqdm

tqdm.pandas()



def check_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        print("路径{}已存在".format(path))

def logger_config(log_path, logging_name):
    '''
    配置log
    :param log_path: 输出log路径
    :param logging_name: 记录中name，可随意
    :return:
    '''
    '''
    logger是日志对象，handler是流处理器，console是控制台输出（没有console也可以，将不会在控制台输出，会在日志文件中输出）
    '''
    # 获取logger对象,取名
    logger = logging.getLogger(logging_name)
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.INFO)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger



def apply_parallel(dfGrouped, func):
    with Parallel(n_jobs=8, backend='multiprocessing', verbose=10) as parallel:
        retLst = parallel(delayed(func)(group) for name, group in dfGrouped)
        result = pd.concat(retLst)
        del retLst
        gc.collect()
        return result


def get_batch_data(data=None, batch_size=None, shuffle=False):
    """
    产生批数据
    :param data: 输入数据 列表
    :param batch_size: 批数据大小
    :param shuffle: 是否打乱数据
    :return:
    """
    rows = len(data)  # 数据条数
    indices = list(range(rows))
    # 是否打乱
    if shuffle:
        random.seed(2020)
        random.shuffle(indices)
    while True:
        batch_indices = np.asarray(indices[0:batch_size])
        indices = indices[batch_size:] + indices[:batch_size]

        print(indices)

        data = np.asarray(data)
        temp_data = data[batch_indices]
        yield temp_data.tolist()


def gen_batch_data(data=None, batch_size=32):
    """
    生成batch list数据
    :param data:
    :param batch_size:
    :return:
    """
    l = len(data)
    for ndx in range(0, l, batch_size):
        yield data[ndx:min(ndx + batch_size, l)]


def find_lcsubstr(s1, s2):
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]  # 生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax = 0  # 最长匹配的长度
    p = 0  # 最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax  # 返回最长子串及其长度


# print(find_lcsubstr('香港疫情','疫情'))


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))

    return df
