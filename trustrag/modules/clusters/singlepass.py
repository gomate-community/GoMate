#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:quincy qiang
# email:yanqiangmiffy@gmail.com
# datetime:2021/5/25 14:32
# description:"do something"
import collections
import os
import warnings
from collections import Counter

import jieba
import jieba.posseg
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from trustrag.modules.clusters.libraries.display import usual_print
from trustrag.modules.clusters.representations.tfidf import TfidfVector


class SGCluster(object):
    def __init__(self,
                 vector_path=None,
                 result_txt_file=None,
                 output_file=None,
                 threshold=0.3,
                 max_features=88888,
                 n_components=1024,
                 ngrams=2,
                 level=1,
                 ):
        self.vector_file = vector_path
        self.result_path = result_txt_file
        self.output_file = output_file
        self.threshold = threshold
        self.max_features = max_features
        self.n_components = n_components
        self.ngrams = ngrams
        self.level=level

    def tokenize(self, text):
        """
        分词
        :param text:
        :return:
        """
        return " ".join([w for w in jieba.cut(text)])

    def cosine(self, rep1, rep2):
        """
        修正后的余弦相似度
        :param rep1: numpy.array([*,*,*,...])
        :param rep2: numpy.array([*,*,*,...])
        :return: float
        """
        assert rep1.shape == rep2.shape
        cos = np.sum(np.multiply(rep1, rep2)) / (np.linalg.norm(rep1) * np.linalg.norm(rep2))
        return cos

    def get_max_similarity(self, dic_topic=None, vector=None, is_avg=False):
        """
        获取文档相似度最大的聚类中心
        :param dic_topic:
        :param vector:
        :param is_avg:
        :return:
        """
        max_index = -1
        max_value = 0
        for k, cluster in dic_topic.items():
            if is_avg:
                # 利用当前簇所有文档与候选文档计算相似度，之后得到平均相似度
                one_similarity = np.mean([self.cosine(vector, v) for v in cluster])
            else:
                # 利用当前簇的平均向量计算与候选文档的相似度
                mean_v = np.mean(cluster, axis=0)
                one_similarity = self.cosine(mean_v, vector)
            if one_similarity > max_value:
                max_value = one_similarity
                max_index = k
        return max_index, max_value

    def get_keywords(slef, cluster_text):
        """
        获取聚类关键词
        :param cluster_text:
        :return:
        """
        sentence_seged = jieba.posseg.cut(cluster_text.strip())
        pos = ['Ag', 'an', 'i', 'Ng', 'n', 'nr', 'ns', 'nt', 'nz', 'v', 'eng']
        keywords = [x.word for x in sentence_seged if x.flag in pos and len(x.word) > 1]
        keywords = [word for word in keywords if not word.endswith('报')]
        word_cnt = Counter(keywords).most_common(3)
        keywords = [w[0] for w in word_cnt]
        keywords = ",".join(keywords)
        return keywords

    def generate_clusters(self, corpus_vectors, text2index, theta):
        clusters = {}
        cluster_text = {}
        num_topic = 0

        for vector, text in tqdm(zip(corpus_vectors, text2index),
                                 total=len(corpus_vectors),
                                 desc="single-pass clustering..."):
            if num_topic == 0:  # 选取第一个文档
                clusters.setdefault(num_topic, []).append(vector)
                cluster_text.setdefault(num_topic, []).append(text)
                num_topic += 1
            else:
                max_index, max_value = self.get_max_similarity(clusters, vector, False)
                if max_value > theta:
                    clusters[max_index].append(vector)
                    cluster_text[max_index].append(text)
                else:  # 创建一个新簇
                    clusters.setdefault(num_topic, []).append(vector)
                    cluster_text.setdefault(num_topic, []).append(text)
                    num_topic += 1
        return clusters, cluster_text,

    def classify(self, data):
        if 'text' not in data:
            data['text'] = (data['title'].astype(str) + ' ' + data['content'].astype(str)).apply(
                lambda x: self.tokenize(x))
        tv = TfidfVector()
        corpus_vectors = tv.online_transform(corpus=data['text'], max_features=self.max_features,
                                                 n_components=self.n_components, is_svd=True, ngrams=(1, self.ngrams))
        # print(corpus_vectors)
        print("矩阵大小：",corpus_vectors.shape)
        np.save(self.vector_file, corpus_vectors)

        index2ids = collections.OrderedDict()
        index2corpus = collections.OrderedDict()
        for index, line in data.iterrows():
            index2ids[index] = line['id']
            index2corpus[index] = line['title']
        print(len(set(index2ids.values())))
        text2index = list(index2corpus.keys())
        print('docs total size:{}'.format(len(text2index)))

        clusters, cluster_text = self.generate_clusters(corpus_vectors,
                                                        text2index,
                                                        theta=self.threshold)
        print("." * 30)
        print("得到的类数量有: {} 个 ...".format(len(clusters)))
        print("." * 30)

        # 按聚类语句数量对聚类结果进行降序排列
        clusterTopic_list = sorted(cluster_text.items(), key=lambda x: len(x[1]), reverse=True)
        csv_data = []
        artilceid_clusterid = {}
        clusterid_keywords = {}
        with open(self.result_path, 'w', encoding='utf-8') as file_write:
            for k in clusterTopic_list:
                cluster_text = []
                for index, value in enumerate(k[1], start=1):
                    cluster_text.append(
                        '(' + str(index) + '): ' + str(index2corpus[value]) + '\t' + str(index2ids[value]))
                    csv_data.append([k[0], len(k[1]), str(index2corpus[value]), str(index2ids[value])])
                    artilceid_clusterid[str(index2ids[value])] = k[0]
                cluster_text = '\n'.join(cluster_text)
                keywords = self.get_keywords(cluster_text)
                clusterid_keywords[k[0]] = keywords
                file_write.write(
                    "【关键词】：{}\n【簇索引】:{} \n【簇中文档数】：{} \n【簇中文档】 ：\n{} \n".format(keywords, k[0], len(k[1]),
                                                                                              cluster_text))
                file_write.write('\n')
                file_write.flush()
        if self.level==1:

            print("len(artilceid_clusterid)", len(artilceid_clusterid))
            print("len(clusterid_keywords)", len(clusterid_keywords))
            # print(artilceid_clusterid)
            # print(clusterid_keywords)
            data['cluster_index'] = data['id'].map(artilceid_clusterid)
            data['cluster_label'] = data['cluster_index'].map(clusterid_keywords)

            data['cluster_count'] = data.groupby(by='cluster_index')['id'].transform('count')
            usual_print(self.output_file, "正在保存到")
            # print(data)
            save_cols = ['id', 'title', 'content','url', 'cluster_index', 'cluster_label', 'cluster_count']
        else:
            print("len(artilceid_clusterid)", len(artilceid_clusterid))
            print("len(clusterid_keywords)", len(clusterid_keywords))
            data['cluster_level1_index'] = data['cluster_index']
            data['cluster_level2_index'] = data['id'].map(artilceid_clusterid)
            data['cluster_label'] = data['cluster_index'].map(clusterid_keywords)

            data['cluster_count'] = data.groupby(by='cluster_index')['id'].transform('count')
            usual_print(self.output_file, "正在保存到")
            # print(data)
            save_cols = ['id', 'title', 'content', 'url','cluster_level1_index', 'cluster_level2_index', 'cluster_label',
                         'cluster_count']
        if self.output_file.endswith('xlsx'):
            data[save_cols].to_excel(self.output_file, index=None)
        elif self.output_file.endswith('csv'):
            data[save_cols].to_csv(self.output_file, index=None)
        elif self.output_file.endswith('json'):
            with open(self.output_file, 'w', encoding='utf-8') as file:
                data[save_cols].to_json(file, orient="records",
                                        lines=True,
                                        force_ascii=False)
        else:
            data[save_cols].to_excel(self.output_file, index=None)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='请输入聚类程序参数')
    # parser.add_argument('--vector_path', type=str, required=False,default="vector.npy", help='文档中间向量文件')
    # parser.add_argument('--result_txt_file', type=str, required=False,default="result.txt", help='聚类结果记录文件')
    # parser.add_argument('--output_file', type=str, required=False, default="result.xlsx",help='聚类结果输出文件 csv/xlsx/json/etc')
    # parser.add_argument('--threshold', type=float, default=0.20,
    #                     help='文档之间合并的相似度阈值，该值越大产生的聚类数量越多')
    # parser.add_argument('--max_features', type=float, default=88888,
    #                     help='Tfidf模型词汇表大小，可以根于输入的文档规模调整')
    # parser.add_argument('--n_components', type=float, default=1024, help='文档向量维度')
    # parser.add_argument('--ngrams', type=float, default=2, help='ngram大小，避免维度稀疏，建议1-3')
    # args = parser.parse_args()

    # file_name = 'result01_2021-06-02_rules.xlsx'
    # data = pd.read_excel('data-zh-main.xlsx', dtype={'id': str})
    # data = pd.DataFrame(documents, columns=['id', 'title', 'content'])
    data = pd.read_excel('data.xlsx', dtype={'id': str})
    data = data.drop_duplicates(subset=['title']).reset_index(drop=True)
    print(data.shape)
    data['id'] = data['id'].astype(str)
    sc = SGCluster(
        vector_path="vector.npy",
        result_txt_file="result.txt",
        output_file="result.xlsx",
        threshold=0.4,
        max_features=8888,
        n_components=1024,
        ngrams=2,
    )
    sc.classify(data)
