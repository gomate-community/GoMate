import re
import os
import jieba
from ltp import LTP
import logging
jieba.setLogLevel(logging.INFO)

def REstrip(text):
    # # 去掉首尾的param
    # text0 = ''
    # i = 0
    # remove_nota = u'[’·!"#$%&\'()*+,-./:;<=>?@，。?★、…【】（）《》？“”‘’！[\\]^_`{|}~]+'
    # while text0 != text or i == 0:
    #     i += 1
    #     text0 = text
    #     text = text.strip().strip(string.punctuation)
    #     text = text.strip(remove_nota)
    #     mo = re.compile(r'^([' + str(param) + r']*)(.*?)([' + str(param) + ']*)$')
    #     result = mo.search(text)
    #     text = result.group(2)
    # return text
    if ' ' in text:
        if len(text) > 10:
            text = text.split(' ')[0]
        else:
            text = text
    if '：' in text:
        cands = text.split('：')
        if len(cands) > 1:
            if len(cands[-1]) > 1:
                text = cands[-1]
            else:
                text = text
        else:
            text = cands[0]
    if ':' in text:
        cands = text.split(':')
        if len(cands) > 1:
            text = cands[-1]
        else:
            text = cands[0]
    if '　' in text:
        text = text.split('　')[0]

    # title 预处理
    punc = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
    # text = re.sub(r"[%s]+" % punc, " ", text)
    text = re.sub(u"\\（.*?）|\\{.*?}|\\[.*?]|\\【.*?】", "", text)
    return text


def load_stopwords():
    base_path = os.getcwd().replace('\\', '/')
    # with open(base_path + '/data/stopwords/hit_stopwords_custom.txt', 'r', encoding='utf-8') as f:
    with open(base_path + '/data/stopwords/baidu_stopwords.txt', 'r', encoding='utf-8') as f:
        words = f.read().split('\n')
    return words


# stop_words = load_stopwords()


def remove_blank_lines(input_str):
    """
    去除空行，去除多余空字符，去除\n \t
    :param input_str: 输入文本
    :return:
    """
    # 去除换行符以及多余空白字符
    input_str = ''.join(input_str.split()).strip()
    return input_str


# remove_blank_lines(text)


def remove_news_stops(input_str):
    """
    # 去除新闻报道中常用的文字
    # https: // blog.csdn.net / lzz781699880 / article / details / 105405793
        str = '氯化锂(3项)'
        name = re.sub(\(.*?项\),'',str)
        print(name)    氯化锂
    :param input_str:
    :return:
    """
    pattern1 = re.compile(r'【.*?讯】|（记者.*?）')
    input_str = re.sub(pattern1, '', input_str)
    # pattern2 = re.compile(r'^.*讯|^记者.*报道')
    pattern2 = re.compile(r'^.{0,10}讯|^记者.*报道')
    input_str = re.sub(pattern2, '', input_str)
    input_str = re.sub('栏名:', '', input_str)
    input_str = re.sub('作者:', '', input_str)
    return input_str


# remove_news_stops(text)


def remove_stopwords(input_str):
    words = [w for w in jieba.cut(input_str)]
    stop_words = load_stopwords()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


def remove_punctuation(input_str):
    """
    去除标点符号
    https://zhuanlan.zhihu.com/p/53277723
    :param input_str:
    :return:
    """
    pattern = "[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】「」『』]"
    pattern = re.compile(pattern)
    sentence = re.sub(pattern, '', input_str)
    sentence = sentence.strip()
    return sentence


def text_tokenizer(input_str, remove_stop=True, remove_punc=True):
    """
    实现中文文本分词
    :param input_str: 输入文本
    :param remove_stop: 是否去除停用词
    :param remove_punc: 是否去除标点符号
    :return:
    """
    # input_str = remove_blank_lines(input_str)
    input_str = remove_news_stops(input_str)
    if remove_stop:
        input_str = remove_stopwords(input_str)
    if remove_punc:
        input_str = remove_punctuation(input_str)
    input_str = " ".join([w for w in input_str.split() if w])
    # input_str = " ".join(input_str.split())
    return input_str


# text_tokenizer('A burden for Biden')


article = """【东方日报专讯】【本报讯】中国海警上周日（23日）拘捕十二名人蛇，当中包括早前涉嫌违反《港区国安法》而被捕的「香港故事」成员李宇轩，以及多名涉反修例案件的人士。昨有报道引述消息称，偷渡案的幕后黑手怀疑是一名台北牧师，并称该牧师与本港违法占领行动（占中）其中一名发起人朱耀明相熟。朱发表声明，否认有参与该十二名人士的偷渡安排，又称并不认识该牧师。
称不认识涉案台北牧师朱耀明昨透过声明表示，有关报道涉及其本人的内容完全失实，又批评该报道暗示他参与安排偷渡计划，做法居心叵测。朱重申并不认识该牧师，亦从未与他见面或以其他方式联系，而其本人无参与该十二名人士的偷渡安排。
涉及计划偷渡到台湾而被捕的十二人中，「香港故事」成员李宇轩于本月十日被捕，当日警方国家安全处亦以涉违《港区国安法》为由，拘捕包括壹传媒黎智英等多人，并于翌日通缉朱耀明之子、「香港民主委员会」总监朱牧民。
此外，朱牧民亦早于七月三十一日，联同逃亡英国的前香港众志常委罗冠聪等另外五人，因涉违《港区国安法》而被通缉。"""


def gen_first_para(input_str, text_length=None):
    """
    输入一篇文章内容 获取第一段
    :param text_length: 目标文本长度
    :param input_str: 输入文章内容
    :return:
    """
    ltp = LTP()

    input_str = text_tokenizer(input_str)
    sents = ltp.sent_split([input_str])
    final_str = ''
    if text_length:
        for sent in sents:
            if len(final_str) < text_length:
                final_str += sent
            else:
                break
    else:
        final_str = ''.join(sents)

    # sents = input_str.split(' ')
    # final_str = remove_punctuation(final_str)
    print(final_str)
    print("==============" * 20 + "\n")
    return final_str

# gen_first_para(article)
# remove_punctuation(article)
