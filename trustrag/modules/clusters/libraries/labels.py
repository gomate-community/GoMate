#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:quincy qiang
# email:yanqiangmiffy@gmail.com
# datetime:2021/5/25 15:15
# description:"do something"
import os

import pandas as pd
from tqdm import tqdm
from libraries.utils import find_lcsubstr


def load_corpus():
    corpus_dir = './data/month/csv/'
    tmp = []
    for csv in os.listdir(corpus_dir):
        csv_df = pd.read_csv(corpus_dir + '/' + csv, error_bad_lines=False)
        tmp.append(csv_df)
    df = pd.concat(tmp, axis=0)

    df['label'] = df['label'].apply(lambda x: "".join(x.split('、')[1:]))
    past_labels = []
    for label in list(set(df['label'])):
        past_labels.append(label)
    return df, past_labels


def load_noduplicated_labels(data):
    # print("构建历史相似label...")
    # data, _ = load_corpus()

    # 将 港报其他要闻 替换成 港报要闻
    data['label'] = data['label'].apply(lambda x: x.replace('港报要闻', '港报其他要闻') if '港报要闻' in x else x)
    data['label'] = data['label'].apply(lambda x: x.split('_')[0] + '_中央及特区政府' if '特区政府' in x else x)
    # 合并相似label
    tmp = pd.DataFrame(data.label.value_counts())
    # print(tmp)
    tmp['value'] = tmp['label']
    tmp['label'] = tmp.index.values.tolist()
    tmp['label1'] = tmp['label'].apply(lambda x: x.split('_')[0])
    tmp['label2'] = tmp['label'].apply(lambda x: x.split('_')[1])
    tmp = tmp.reset_index(drop=True)
    # print(tmp)
    # ('非常任法官施觉民辞', 9)
    new_tmp = tmp.copy()
    match_res = []
    for index, row in tmp.iterrows():
        tmp_res = set()
        tmp_res.add(row.label)
        for new_index, new_row in new_tmp.iterrows():
            try:
                # if row.label == '新冠肺炎_疫情' and new_row.label == '新冠肺炎_香港疫情':
                #     print(row.label, new_row.label, row.label1 == new_row.label1)
                if row.label1 == new_row.label1:
                    if row.label2[0] == new_row.label2[0]:  # 判断label的开头是否相同
                        res = find_lcsubstr(row.label2, new_row.label2)
                        if res[1] >= 3:
                            tmp_res.add(new_row.label)
                    elif len(row.label2) <= 2 or len(new_row.label2) <= 2:
                        # 新冠肺炎_香港疫情 新冠肺炎_疫情|修例风波_检控 修例风波_私人检控|港报要闻_港大 港报要闻_香港大学
                        res = find_lcsubstr(row.label2, new_row.label2)
                        if res[0] == new_row.label2 or res[0] == row.label2:
                            # print(row.label, new_row.label)
                            tmp_res.add(new_row.label)
            except Exception as e:
                print(row)
        match_res.append(sorted(tmp_res))
    tmp['match_label'] = match_res
    existed_label = tmp['match_label'].values.tolist()

    def is_contained(label):
        for ex in existed_label:
            if set(ex) & set(label):
                if ex != label and len(ex) > len(label):
                    return ex
        return label

    tmp['final_label'] = tmp['match_label'].apply(lambda x: is_contained(x))
    tmp['final_label'] = tmp['final_label'].apply(lambda x: x[-1])
    # print(tmp[['final_label','match_label']])
    # print(tmp['final_label'].nunique())
    tmp.to_csv('data/outputs/labels/labels_dropdup.csv', index=False)
    label_dic = dict(zip(tmp['label'], tmp['final_label']))
    return label_dic


def drop_duplicated_labels(data):
    print("========================")
    print("去除label之前文章label个数：", data['label'].nunique())
    label_dict = load_noduplicated_labels(data)
    data['label'] = data['label'].apply(lambda x: label_dict[x] if x.strip() in label_dict else x)
    # data['final_label'] = data['label'].map(label_dict) # 会导致有些label为空
    # print(data.shape[0] - data.count())
    print("去除label之后文章label个数：", data['label'].nunique())
    print("========================")
    # print(data.shape[0] - data.count())
    return data
