#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:quincy qiang
# datetime:2021/5/11 16:26
# description:"tfidf模型聚类与分类"
import logging
import os
import time

import jieba
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import loguru
jieba.setLogLevel(logging.INFO)


class TfidfVector(object):
    def __init__(self):
        pass

    def online_transform(self,
                         corpus=None,
                         max_features=88888,
                         n_components=1024,
                         is_svd=True, ngrams=(1, 1)
                         ):
        """
        实时训练模型并进行返回转换向量
        :param corpus: 分完词的用于训练tfidf模型的语料
        :param max_features:
        :param ngrams:
        :param is_svd: 是否进行降维
        :param n_components:
        :param input_docs: 需要转化的语料
        :return:
        """
        vec = TfidfVectorizer(max_features=max_features, ngram_range=ngrams,
                              min_df=2, max_df=0.96,
                              strip_accents='unicode',
                              norm='l2',
                              token_pattern=r"(?u)\b\w+\b")

        X = vec.fit_transform(corpus)  # sparse matrix
        if not is_svd:
            X = X.toarray()
        print("原始的tfidf矩阵大小X.shape", X.shape)
        print("词汇量：", len(vec.vocabulary_))  # 词汇量： 76381
        # print(vec.idf_)
        # print(vec.vocabulary_)
        if is_svd:
            t0 = time.time()
            try:
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                normalizer = Normalizer(copy=False)
                lsa = make_pipeline(svd, normalizer, verbose=True)
                X = lsa.fit_transform(X)

            except ValueError as e:
                if "n_components" in str(e):
                    # 获取实际特征数量
                    n_features = X.shape[1]
                    loguru.logger.warning(f"n_components({n_components})大于特征数量({n_features})，自动调整为{n_features - 1}")

                    # 重新设置组件数量为特征数量-1
                    adjusted_n_components = n_features - 1
                    svd = TruncatedSVD(n_components=adjusted_n_components, random_state=42)
                    normalizer = Normalizer(copy=False)
                    lsa = make_pipeline(svd, normalizer, verbose=True)
                    X = lsa.fit_transform(X)
                else:
                    # 如果是其他ValueError，则向上传递
                    raise e
                # 打印结果
            print("通过SVD降维之后的X.shape", X.shape)
            print("done in %fs" % (time.time() - t0))

            explained_variance = svd.explained_variance_ratio_.sum()
            print("Explained variance of the SVD step: {}%".format(
                int(explained_variance * 100)))

        return X


if __name__ == '__main__':
    date = '2021-05-10'
    file_name = '/home/edit/newspaper/news_recommend/result/{}/result01_{}.xlsx'.format(date, date)
    data = pd.read_excel(file_name)

    tv = TfidfVector()

    corpus_vectors = tv.online_transform(data['text'], is_svd=False, ngrams=(1, 1))
