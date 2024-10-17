import os
import re
import json
import math
import numpy as np
from gensim import corpora, models, similarities, matutils
from smart_open import smart_open
import pandas as pd


class Single_Pass_Cluster(object):

    def __init__(self,
                 filename,
                 stop_words_file='停用词汇总.txt',
                 theta=0.5,
                 LTP_DATA_DIR=r'D:\ltp-models\\',  # ltp模型目录的路径

                 ):

        self.filename = filename
        self.stop_words_file = stop_words_file
        self.theta = theta
        self.LTP_DATA_DIR = LTP_DATA_DIR
        self.cws_model_path = os.path.join(self.LTP_DATA_DIR, 'cws.model')
        self.pos_model_path = os.path.join(self.LTP_DATA_DIR, 'pos.model')

    def loadData(self, filename):
        pass




    def get_Doc2vec_vector_representation(self, word_segmentation):
        # 得到文本数据的空间向量表示

        corpus_doc2vec = [get_avg_feature_vector(i, model, num_features=50) for i in word_segmentation]
        return corpus_doc2vec

    def getMaxSimilarity(self, dictTopic, vector):
        maxValue = 0
        maxIndex = -1
        for k, cluster in dictTopic.items():
            oneSimilarity = np.mean([matutils.cossim(vector, v) for v in cluster])
            # oneSimilarity = np.mean([cosine_similarity(vector, v) for v in cluster])
            if oneSimilarity > maxValue:
                maxValue = oneSimilarity
                maxIndex = k
        return maxIndex, maxValue

    def single_pass(self, corpus, texts, theta):
        dictTopic = {}
        clusterTopic = {}
        numTopic = 0
        cnt = 0
        for vector, text in zip(corpus, texts):
            if numTopic == 0:
                dictTopic[numTopic] = []
                dictTopic[numTopic].append(vector)
                clusterTopic[numTopic] = []
                clusterTopic[numTopic].append(text)
                numTopic += 1
            else:
                maxIndex, maxValue = self.getMaxSimilarity(dictTopic, vector)
                # 将给定语句分配到现有的、最相似的主题中
                if maxValue >= theta:
                    dictTopic[maxIndex].append(vector)
                    clusterTopic[maxIndex].append(text)

                # 或者创建一个新的主题
                else:
                    dictTopic[numTopic] = []
                    dictTopic[numTopic].append(vector)
                    clusterTopic[numTopic] = []
                    clusterTopic[numTopic].append(text)
                    numTopic += 1
            cnt += 1
            if cnt % 500 == 0:
                print("processing {}...".format(cnt))
        return dictTopic, clusterTopic

    def fit_transform(self, theta=0.5):
        datMat = self.loadData(self.filename)
        word_segmentation = []
        for i in range(len(datMat)):
            word_segmentation.append(self.word_segment(datMat[i]))
        print("............................................................................................")
        print('文本已经分词完毕 !')

        # 得到文本数据的空间向量表示
        corpus_tfidf = self.get_Doc2vec_vector_representation(word_segmentation)
        # corpus_tfidf =  self.get_Doc2vec_vector_representation(word_segmentation)
        dictTopic, clusterTopic = self.single_pass(corpus_tfidf, datMat, theta)
        print("............................................................................................")
        print("得到的主题数量有: {} 个 ...".format(len(dictTopic)))
        print("............................................................................................\n")
        # 按聚类语句数量对主题进行排序，找到重要的聚类群
        clusterTopic_list = sorted(clusterTopic.items(), key=lambda x: len(x[1]), reverse=True)
        print(clusterTopic_list)