import json
import re
from typing import List

import jieba

from gomate.modules.document.utils import PROJECT_BASE


class SourceCitation:
    def __init__(self):
        self.stopwords = ["的"]

    # def cut(self, para: str):
    #     return [x for x in para.split("。") if x]

    def cut(self, para: str):

        # 定义结束符号列表
        end_symbols = ['。', '！', '？', '…', '；', '\n']

        # 定义引号对
        quote_pairs = {'"': '"', "'": "'", '「': '」', '『': '』'}

        sentences = []
        current_sentence = ''
        quote_stack = []

        for char in para:
            current_sentence += char

            # 处理引号
            if char in quote_pairs.keys():
                quote_stack.append(char)
            elif quote_stack and char in quote_pairs.values():
                if char == quote_pairs[quote_stack[-1]]:
                    quote_stack.pop()

            # 当遇到结束符号且不在引号内时，进行分句
            if char in end_symbols and not quote_stack:
                # 去除可能的空白符号
                # sentence = current_sentence.strip()
                sentence = current_sentence
                if sentence:
                    sentences.append(sentence)
                current_sentence = ''

        # 处理末尾可能剩余的文本
        if current_sentence:
            sentences.append(current_sentence)

        return sentences

    def remove_stopwords(self, query: str):
        for word in self.stopwords:
            query = query.replace(word, " ")
        return query

    def load_response_json(self, response):
        cleaned_response = re.sub(r'^.*?```json\n|```$', '', response, flags=re.DOTALL)
        print(cleaned_response)
        data = json.loads(cleaned_response)
        return data

    def deduplicate_docs(self, docs):
        new_docs = []
        is_exits = []
        for doc in docs:
            if doc['content'] not in is_exits:
                is_exits.append(doc['content'])
                new_docs.append(doc)

    def convert_to_chinese(self, number_str):
        # 定义单个数字到汉字的映射
        digit_to_chinese = {
            '0': '零',
            '1': '一',
            '2': '二',
            '3': '三',
            '4': '四',
            '5': '五',
            '6': '六',
            '7': '七',
            '8': '八',
            '9': '九'
        }
        number = int(number_str)  # 将输入的字符串转换为整数
        if number == 0:
            return digit_to_chinese['0']  # 直接处理 0 的情况
        result = ""

        # 处理 10 到 99 的数字
        if number >= 10 and number < 100:
            tens = number // 10  # 获取十位
            ones = number % 10  # 获取个位

            # 处理十位数
            if tens > 1:
                result += digit_to_chinese[str(tens)]  # 如果十位大于 1，需要显示数字
            result += '十'  # 始终加上 "十" 表示十位

            # 处理个位数
            if ones > 0:
                result += digit_to_chinese[str(ones)]
        else:
            # 处理个位数 (1-9)
            result += digit_to_chinese[number_str]

        return result

    # def format_text_data(self,data):
    #     formatted_text = ""
    #     for item in data:
    #         formatted_text += f"\n{item['title']}\n{item['content']}\n\n{item['source']}\n"
    #     return formatted_text.strip()

    # def format_text_data(self,data):
    #     formatted_text = ""
    #     for item in data:
    #         formatted_text += f"```\n{item['title']}\n{item['content']}\n\n{item['source']}\n```\n\n"
    #     return formatted_text.strip()

    def format_text_data(self, data):
        formatted_text = ""
        for i, item in enumerate(data):
            if i > 0:
                formatted_text += "---\n\n"  # Add Markdown horizontal rule between groups
            formatted_text += f"```\n{item['title']}\n{item['content']}\n\n{item['source']}\n```\n\n"
        return formatted_text.strip()

    def ground_response(
            self,
            question: str,
            response: str,
            evidences: List[str],
            selected_idx: List[int],
            markdown: bool = True,
            show_code=False,
            selected_docs=List[dict]
    ):

        # Create JSON object
        json_data = {
            "question": question,
            "response": response,
            "evidences": evidences,
            "selected_idx": selected_idx,
            "selected_docs": selected_docs
        }
        response = self.load_response_json(response)
        print(response)
        contents = response['contents']
        for cit_idx, citation in enumerate(contents):
            sentence = citation['title'] + citation['content']
            sentence_seg_cut = set(jieba.lcut(self.remove_stopwords(sentence)))
            sentence_seg_cut_length = len(sentence_seg_cut)

            best_score = 0.0
            best_idx = -1
            best_sentence = ''
            for doc_idx, doc in enumerate(selected_docs):
                evidence_sentences = self.cut(doc['content'])
                for es_idx, evidence_sentence in enumerate(evidence_sentences):
                    evidence_seg_cut = set(jieba.lcut(self.remove_stopwords(evidence_sentence)))
                    overlap = sentence_seg_cut.intersection(evidence_seg_cut)
                    ratio = len(overlap) / sentence_seg_cut_length
                    if ratio > best_score:
                        best_score = ratio
                        best_idx = doc_idx
                        best_sentence = evidence_sentence
            citation['title'] = self.convert_to_chinese(str(cit_idx + 1)) + '、' + citation['title']
            citation['content'] = best_sentence
            citation['best_idx'] = best_idx
            citation['best_score'] = best_score
            # citation['newsinfo_title'] = selected_docs[best_idx]['newsinfo']['title']
            # citation['newsinfo_date'] = selected_docs[best_idx]['newsinfo']['date']
            # citation['newsinfo_source'] = selected_docs[best_idx]['newsinfo']['source']
            newsinfo = selected_docs[best_idx]['newsinfo']
            citation['source'] = '--- ' + newsinfo['title'] + ' ' + newsinfo['date'] + ' ' + newsinfo['source']
        citation_content = self.format_text_data(contents)
        response['result'] = citation_content
        response['quote_list'] = []
        print(response['result'])
        return response


if __name__ == '__main__':
    mc = SourceCitation()

    with open(f'{PROJECT_BASE}/data/docs/citations_samples/sample17.json', 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    print(input_data)
    result = mc.ground_response(
        question=input_data["question"],
        response=input_data["response"],
        evidences=input_data["evidences"],
        selected_idx=input_data["selected_idx"],
        markdown=True,
        show_code=True,
        selected_docs=input_data["selected_docs"],
    )

    print(result)
