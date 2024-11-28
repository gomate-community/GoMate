#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: rag_tokenizer.py
@time: 2024/06/06
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import copy
import math
import os
import re
import string
import sys

import datrie
from hanziconv import HanziConv
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# from trustrag.modules.document.utils import get_project_base_directory

_get_module_path = lambda path: os.path.normpath(os.path.join(os.getcwd(),
                                                              os.path.dirname(__file__), path))
DEFAULT_IDF = _get_module_path("huqie.txt")
DEFAULT_IDF_TRIE = _get_module_path("huqie.txt.trie")

print("RAG Vocab:", DEFAULT_IDF)
print("RAG Trie:", DEFAULT_IDF_TRIE)


class RagTokenizer:
    def key_(self, line):
        return str(line.lower().encode("utf-8", 'ignore'))[2:-1]

    def rkey_(self, line):
        return str(("DD" + (line[::-1].lower())).encode("utf-8", 'ignore'))[2:-1]

    def loadDict_(self, fnm):
        print("[HUQIE]:Build trie", fnm, file=sys.stderr)
        try:
            of = open(fnm, "r", encoding='utf-8')
            while True:
                line = of.readline()
                if not line:
                    break
                line = re.sub(r"[\r\n]+", "", line)
                line = re.split(r"[ \t]", line)
                k = self.key_(line[0])
                F = int(math.log(float(line[1]) / self.DENOMINATOR) + .5)
                if k not in self.trie_ or self.trie_[k][0] < F:
                    self.trie_[self.key_(line[0])] = (F, line[2])
                self.trie_[self.rkey_(line[0])] = 1
            self.trie_.save(fnm + ".trie")
            of.close()
        except Exception as e:
            print("[HUQIE]:Faild to build trie, ", fnm, e, file=sys.stderr)

    def __init__(self, debug=False):
        self.DEBUG = debug
        self.DENOMINATOR = 1000000
        self.trie_ = datrie.Trie(string.printable)
        # self.DIR_ = os.path.join(get_project_base_directory(), "rag/res", "huqie")
        # self.DIR_ = os.path.join(get_project_base_directory(), "data/huqie", "huqie")

        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        self.SPLIT_CHAR = r"([ ,\.<>/?;'\[\]\\`!@#$%^&*\(\)\{\}\|_+=《》，。？、；‘’：“”【】~！￥%……（）——-]+|[a-z\.-]+|[0-9,\.-]+)"
        try:
            self.trie_ = datrie.Trie.load(DEFAULT_IDF_TRIE)
            return
        except Exception as e:
            print("[HUQIE]:Build default trie", file=sys.stderr)
            self.trie_ = datrie.Trie(string.printable)

        self.loadDict_(DEFAULT_IDF)

    def loadUserDict(self, fnm):
        try:
            self.trie_ = datrie.Trie.load(DEFAULT_IDF_TRIE)
            return
        except Exception as e:
            self.trie_ = datrie.Trie(string.printable)
        self.loadDict_(fnm)

    def addUserDict(self, fnm):
        self.loadDict_(fnm)

    def _strQ2B(self, ustring):
        """把字符串全角转半角"""
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xfee0
            if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
                rstring += uchar
            else:
                rstring += chr(inside_code)
        return rstring

    def _tradi2simp(self, line):
        return HanziConv.toSimplified(line)

    def dfs_(self, chars, s, preTks, tkslist):
        MAX_L = 10
        res = s
        # if s > MAX_L or s>= len(chars):
        if s >= len(chars):
            tkslist.append(preTks)
            return res

        # pruning
        S = s + 1
        if s + 2 <= len(chars):
            t1, t2 = "".join(chars[s:s + 1]), "".join(chars[s:s + 2])
            if self.trie_.has_keys_with_prefix(self.key_(t1)) and not self.trie_.has_keys_with_prefix(
                    self.key_(t2)):
                S = s + 2
        if len(preTks) > 2 and len(
                preTks[-1][0]) == 1 and len(preTks[-2][0]) == 1 and len(preTks[-3][0]) == 1:
            t1 = preTks[-1][0] + "".join(chars[s:s + 1])
            if self.trie_.has_keys_with_prefix(self.key_(t1)):
                S = s + 2

        ################
        for e in range(S, len(chars) + 1):
            t = "".join(chars[s:e])
            k = self.key_(t)

            if e > s + 1 and not self.trie_.has_keys_with_prefix(k):
                break

            if k in self.trie_:
                pretks = copy.deepcopy(preTks)
                if k in self.trie_:
                    pretks.append((t, self.trie_[k]))
                else:
                    pretks.append((t, (-12, '')))
                res = max(res, self.dfs_(chars, e, pretks, tkslist))

        if res > s:
            return res

        t = "".join(chars[s:s + 1])
        k = self.key_(t)
        if k in self.trie_:
            preTks.append((t, self.trie_[k]))
        else:
            preTks.append((t, (-12, '')))

        return self.dfs_(chars, s + 1, preTks, tkslist)

    def freq(self, tk):
        k = self.key_(tk)
        if k not in self.trie_:
            return 0
        return int(math.exp(self.trie_[k][0]) * self.DENOMINATOR + 0.5)

    def tag(self, tk):
        k = self.key_(tk)
        if k not in self.trie_:
            return ""
        return self.trie_[k][1]

    def score_(self, tfts):
        B = 30
        F, L, tks = 0, 0, []
        for tk, (freq, tag) in tfts:
            F += freq
            L += 0 if len(tk) < 2 else 1
            tks.append(tk)
        F /= len(tks)
        L /= len(tks)
        if self.DEBUG:
            print("[SC]", tks, len(tks), L, F, B / len(tks) + L + F)
        return tks, B / len(tks) + L + F

    def sortTks_(self, tkslist):
        res = []
        for tfts in tkslist:
            tks, s = self.score_(tfts)
            res.append((tks, s))
        return sorted(res, key=lambda x: x[1], reverse=True)

    def merge_(self, tks):
        patts = [
            (r"[ ]+", " "),
            (r"([0-9\+\.,%\*=-]) ([0-9\+\.,%\*=-])", r"\1\2"),
        ]
        # for p,s in patts: tks = re.sub(p, s, tks)

        # if split chars is part of token
        res = []
        tks = re.sub(r"[ ]+", " ", tks).split(" ")
        s = 0
        while True:
            if s >= len(tks):
                break
            E = s + 1
            for e in range(s + 2, min(len(tks) + 2, s + 6)):
                tk = "".join(tks[s:e])
                if re.search(self.SPLIT_CHAR, tk) and self.freq(tk):
                    E = e
            res.append("".join(tks[s:E]))
            s = E

        return res

    def maxForward_(self, line):
        res = []
        s = 0
        while s < len(line):
            e = s + 1
            t = line[s:e]
            while e < len(line) and self.trie_.has_keys_with_prefix(
                    self.key_(t)):
                e += 1
                t = line[s:e]

            while e - 1 > s and self.key_(t) not in self.trie_:
                e -= 1
                t = line[s:e]

            if self.key_(t) in self.trie_:
                res.append((t, self.trie_[self.key_(t)]))
            else:
                res.append((t, (0, '')))

            s = e

        return self.score_(res)

    def maxBackward_(self, line):
        res = []
        s = len(line) - 1
        while s >= 0:
            e = s + 1
            t = line[s:e]
            while s > 0 and self.trie_.has_keys_with_prefix(self.rkey_(t)):
                s -= 1
                t = line[s:e]

            while s + 1 < e and self.key_(t) not in self.trie_:
                s += 1
                t = line[s:e]

            if self.key_(t) in self.trie_:
                res.append((t, self.trie_[self.key_(t)]))
            else:
                res.append((t, (0, '')))

            s -= 1

        return self.score_(res[::-1])

    def english_normalize_(self, tks):
        return [self.stemmer.stem(self.lemmatizer.lemmatize(t)) if re.match(r"[a-zA-Z_-]+$", t) else t for t in tks]

    def tokenize(self, line):
        line = self._strQ2B(line).lower()
        line = self._tradi2simp(line)
        zh_num = len([1 for c in line if is_chinese(c)])
        if zh_num == 0:
            return " ".join([self.stemmer.stem(self.lemmatizer.lemmatize(t)) for t in word_tokenize(line)])

        arr = re.split(self.SPLIT_CHAR, line)
        res = []
        for L in arr:
            if len(L) < 2 or re.match(
                    r"[a-z\.-]+$", L) or re.match(r"[0-9\.-]+$", L):
                res.append(L)
                continue
            # print(L)

            # use maxforward for the first time
            tks, s = self.maxForward_(L)
            tks1, s1 = self.maxBackward_(L)
            if self.DEBUG:
                print("[FW]", tks, s)
                print("[BW]", tks1, s1)

            diff = [0 for _ in range(max(len(tks1), len(tks)))]
            for i in range(min(len(tks1), len(tks))):
                if tks[i] != tks1[i]:
                    diff[i] = 1

            if s1 > s:
                tks = tks1

            i = 0
            while i < len(tks):
                s = i
                while s < len(tks) and diff[s] == 0:
                    s += 1
                if s == len(tks):
                    res.append(" ".join(tks[i:]))
                    break
                if s > i:
                    res.append(" ".join(tks[i:s]))

                e = s
                while e < len(tks) and e - s < 5 and diff[e] == 1:
                    e += 1

                tkslist = []
                self.dfs_("".join(tks[s:e + 1]), 0, [], tkslist)
                res.append(" ".join(self.sortTks_(tkslist)[0][0]))

                i = e + 1

        res = " ".join(self.english_normalize_(res))
        if self.DEBUG:
            print("[TKS]", self.merge_(res))
        return self.merge_(res)

    def fine_grained_tokenize(self, tks):
        tks = tks.split(" ")
        zh_num = len([1 for c in tks if c and is_chinese(c[0])])
        if zh_num < len(tks) * 0.2:
            res = []
            for tk in tks:
                res.extend(tk.split("/"))
            return " ".join(res)

        res = []
        for tk in tks:
            if len(tk) < 3 or re.match(r"[0-9,\.-]+$", tk):
                res.append(tk)
                continue
            tkslist = []
            if len(tk) > 10:
                tkslist.append(tk)
            else:
                self.dfs_(tk, 0, [], tkslist)
            if len(tkslist) < 2:
                res.append(tk)
                continue
            stk = self.sortTks_(tkslist)[1][0]
            if len(stk) == len(tk):
                stk = tk
            else:
                if re.match(r"[a-z\.-]+$", tk):
                    for t in stk:
                        if len(t) < 3:
                            stk = tk
                            break
                    else:
                        stk = " ".join(stk)
                else:
                    stk = " ".join(stk)

            res.append(stk)

        return " ".join(self.english_normalize_(res))


def is_chinese(s):
    if s >= u'\u4e00' and s <= u'\u9fa5':
        return True
    else:
        return False


def is_number(s):
    if s >= u'\u0030' and s <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(s):
    if (s >= u'\u0041' and s <= u'\u005a') or (
            s >= u'\u0061' and s <= u'\u007a'):
        return True
    else:
        return False


def naiveQie(txt):
    tks = []
    for t in txt.split(" "):
        if tks and re.match(r".*[a-zA-Z]$", tks[-1]
                            ) and re.match(r".*[a-zA-Z]$", t):
            tks.append(" ")
        tks.append(t)
    return tks


tokenizer = RagTokenizer()
tokenize = tokenizer.tokenize
fine_grained_tokenize = tokenizer.fine_grained_tokenize
tag = tokenizer.tag
freq = tokenizer.freq
loadUserDict = tokenizer.loadUserDict
addUserDict = tokenizer.addUserDict
tradi2simp = tokenizer._tradi2simp
strQ2B = tokenizer._strQ2B

if __name__ == '__main__':
    tknzr = RagTokenizer(debug=False)
    text = """
    2020年9期总第918期商业研究一、研究背景和意义2020年新春伊始 ，一场突如其来的疫情打乱了人们欢天喜地回家过年的计划 ，节日的气氛被病毒侵肆 ，1月23日，武汉打响“封城”第一枪，接着从湖北各市区乃至全国各地纷纷开 启一级应急响应 ，全国人民停工 、停产、停学,开始居家隔离 ，春节档电影除夕当天宣布延期上映 ，电影院、商场、中小商铺 、外卖、快递几乎全部停工 ，居民消费行为呈现出 “宅经济”的特点，经济社会运行受到严重冲击 。国家统计局数据显示 ，2020年1月-2月，我国社会消费品零售总额同比下降 20.5%。消费不仅是衡量 居民生活水平的重要指标 ，同时亦是我国经济增长的主要驱动 力。因此，新冠疫情在国内得到基本控制的情况下 ，国家一手紧 抓复工复产，一手紧抓内外防控 。有学者认为疫情期间居家隔离举措会影响消费者消费理念，降低整体消费水平 ，也有学者认为 ，疫情结束以后 ，居民会产生“报复性消费 ”，显著提高消费水平 ，更有认为疫情只是暂时 延缓了经济社会发展进程 ，长久看来 ，并不会对消费行为造成较 大影响。研究新冠肺炎疫情下消费者心理 、消费能力 、消费方式 、消费对象等方面的变化有助于判断未来消费行为变动趋势以及 企业应对措施 ，有助于充分挖掘国内市场 、重振居民消费 、全面建成小康社会以及实现 2020年经济社会发展目标 。二、新冠肺炎疫情特点及现状新型冠状病毒肺炎 ，简称“新冠肺炎 ”，2020年2月11日，世界卫生组织其命名为 “COVID-19 ”。
    """
    text = """
    联通云在中国品牌日绽放光芒，引领数字中国新篇章发布时间：2024年5月11日在2024年的璀璨春光中，中国迎来了又一届盛大的中国品牌日活动。这次活动以“中国品牌，世界共享；国货潮牌，共筑未来”为主题，于5月10日至14日在繁华的上海盛大举行。这场由国家发展改革委联合国务院国资委、市场监管总局、国家知识产权局共同主办的盛会，不仅是一次品牌展示的盛宴，更是中国品牌力量向世界发出的强音。中國聯通，作為数字化转型的领军者，携其旗下云计算品牌——联通云，在这场盛宴中精彩亮相，向世界展示了中国联通在智算领域的卓越成就与深远布局。一、中国品牌日：数字经济的时代号角中国品牌日自设立以来，已成为推动我国品牌建设、提升国家品牌竞争力的重要平台。今年，随着数字经济在全球范围内的蓬勃兴起，中国品牌日更是赋予了新的时代内涵。作为数字经济发展的核心驱动力，算力资源的有效配置和高效利用成为关注的焦点。中国联通以此为契机，通过联通云的亮相，向世界展示了其在算力服务领域的深厚积累与创新实践，为数字经济的高质量发展注入了强劲动力。在为期五天的展览中，中国联通展馆成为了吸引无数目光的焦点。这里不仅汇聚了联通云最新的技术成果，更通过一系列生动的场景演示，让观众亲身体验到算力技术如何深刻改变着我们的生活和工作方式。从智慧城市到智能制造，从金融科技到教育医疗，联通云以其卓越的性能和广泛的应用场景，赢得了众多参观者的高度赞誉。二、联通云：数字中国的算力引擎作为服务数字中国云计算的国家队，联通云在中国品牌日的舞台上展现了其强大的技术实力与战略视野。面对数字经济的蓬勃发展，中国联通聚焦网络强国、数字中国两大主责，明确联网通信、算网数智两大主业，以联通云为载体，全面服务数字中国“五位一体”总体布局。
    在这一过程中，联通云充分发挥算网一体的优势，成为构筑多样化算力服务的先行者。
    """
    tks = tknzr.tokenize(text)
    print(tks)
    # print(tknzr.fine_grained_tokenize(tks))
    # # huqie.addUserDict("/tmp/tmp.new.tks.dict")
    # tks = tknzr.tokenize(
    #     "哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈")
    # print(tknzr.fine_grained_tokenize(tks))
    # tks = tknzr.tokenize(
    #     "公开征求意见稿提出，境外投资者可使用自有人民币或外汇投资。使用外汇投资的，可通过债券持有人在香港人民币业务清算行及香港地区经批准可进入境内银行间外汇市场进行交易的境外人民币业务参加行（以下统称香港结算行）办理外汇资金兑换。香港结算行由此所产生的头寸可到境内银行间外汇市场平盘。使用外汇投资的，在其投资的债券到期或卖出后，原则上应兑换回外汇。")
    # print(tknzr.fine_grained_tokenize(tks))
    # tks = tknzr.tokenize(
    #     "多校划片就是一个小区对应多个小学初中，让买了学区房的家庭也不确定到底能上哪个学校。目的是通过这种方式为学区房降温，把就近入学落到实处。南京市长江大桥")
    # print(tknzr.fine_grained_tokenize(tks))
    # tks = tknzr.tokenize(
    #     "实际上当时他们已经将业务中心偏移到安全部门和针对政府企业的部门 Scripts are compiled and cached aaaaaaaaa")
    # print(tknzr.fine_grained_tokenize(tks))
    # tks = tknzr.tokenize("虽然我不怎么玩")
    # print(tknzr.fine_grained_tokenize(tks))
    # tks = tknzr.tokenize("蓝月亮如何在外资夹击中生存,那是全宇宙最有意思的")
    # print(tknzr.fine_grained_tokenize(tks))
    # tks = tknzr.tokenize(
    #     "涡轮增压发动机num最大功率,不像别的共享买车锁电子化的手段,我们接过来是否有意义,黄黄爱美食,不过，今天阿奇要讲到的这家农贸市场，说实话，还真蛮有特色的！不仅环境好，还打出了")
    # print(tknzr.fine_grained_tokenize(tks))
    # tks = tknzr.tokenize("这周日你去吗？这周日你有空吗？")
    # print(tknzr.fine_grained_tokenize(tks))
    # tks = tknzr.tokenize("Unity3D开发经验 测试开发工程师 c++双11双11 985 211 ")
    # print(tknzr.fine_grained_tokenize(tks))
    # tks = tknzr.tokenize(
    #     "数据分析项目经理|数据分析挖掘|数据分析方向|商品数据分析|搜索数据分析 sql python hive tableau Cocos2d-")
    # print(tknzr.fine_grained_tokenize(tks))
    # if len(sys.argv) < 2:
    #     sys.exit()
    # tknzr.DEBUG = False
    # tknzr.loadUserDict(sys.argv[1])
    # of = open(sys.argv[2], "r")
    # while True:
    #     line = of.readline()
    #     if not line:
    #         break
    #     print(tknzr.tokenize(line))
    # of.close()
