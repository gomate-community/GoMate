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

    # def format_text_data(self, data):
    #     formatted_text = ""
    #     for i, item in enumerate(data):
    #         if i > 0:
    #             formatted_text += "---\n\n"  # Add Markdown horizontal rule between groups
    #         formatted_text += f"```\n{item['title']}\n{item['content']}\n\n{item['source']}\n```\n\n"
    #     return formatted_text.strip()

    def highlight_common_substrings(self, sentence, evidence_sentence, evidence, min_length=6):
        evidence_sentences = self.cut(evidence)
        current_sentence_index = next(i for i, s in enumerate(evidence_sentences) if evidence_sentence == s)
        highlighted_text = evidence_sentences[current_sentence_index]
        start_evidence = evidence.index(highlighted_text)
        end_evidence = start_evidence + len(highlighted_text)
        return [[start_evidence, end_evidence - 1]]

    def format_text_data(self, data):
        formatted_text = ""
        for i, item in enumerate(data):
            if i > 0:
                formatted_text += "---\n\n"  # Add Markdown horizontal rule between groups
            formatted_text += f"```\n{item['title']}\n{item['content']}\n```\n\n"
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
        contents = [content  for content in response['contents'] if 'title' in content and 'content' in content]

        final_response = []
        quote_list = []
        best_indices = 0

        new_contents = []
        for cit_idx, citation in enumerate(contents):
            sentence = citation['title'] + citation['content']
            sentence_seg_cut = set(jieba.lcut(self.remove_stopwords(sentence)))
            sentence_seg_cut_length = len(sentence_seg_cut)

            best_ratio = 0.0
            best_sentence = ''
            best_idx = 0
            highlighted_start_end = []

            for doc_idx, doc in enumerate(selected_docs):
                evidence_sentences = self.cut(doc['content'])
                for es_idx, evidence_sentence in enumerate(evidence_sentences):
                    evidence_seg_cut = set(jieba.lcut(self.remove_stopwords(evidence_sentence)))
                    overlap = sentence_seg_cut.intersection(evidence_seg_cut)
                    ratio = len(overlap) / sentence_seg_cut_length
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_idx = doc_idx
                        best_sentence = evidence_sentence
                        highlighted_start_end = self.highlight_common_substrings(sentence, evidence_sentence,
                                                                                 doc['content'])

            citation['citation_content'] = best_sentence
            citation['best_idx'] = best_idx
            citation['best_ratio'] = best_ratio
            citation['highlighted_start_end'] = highlighted_start_end
            # citation['title'] = self.convert_to_chinese(str(cit_idx + 1)) + '、' + citation['title']
            # newsinfo = selected_docs[best_idx]['newsinfo']
            # citation['source'] = '--- ' + newsinfo['title'] + ' ' + newsinfo['date'] + ' ' + newsinfo['source']

        # 是否设置标题序号：如果存在多个段落需要设置，如果只有一个段落那么不需要设置
        citation_cnt = 0

        is_citation_exists = []
        for citation in contents:
            best_idx = citation['best_idx']
            if best_idx not in is_citation_exists:
                new_contents.append(citation)
                is_citation_exists.append(best_idx)
                citation_cnt += 1

        is_content_exists = []

        for citation in contents:
            group_list = []
            best_idx = citation['best_idx']
            if best_idx not in is_content_exists:
                if citation_cnt > 1:
                    citation['title'] = self.convert_to_chinese(str(best_indices + 1)) + '、' + citation['title']
                    citation['title'] = "**" + citation['title'] + "**"
                else:
                    citation['title'] = "**" + citation['title'] + "**"
                new_contents.append(citation)

                group_item = {
                    "doc_id": selected_docs[best_idx]["doc_id"],
                    "chk_id": selected_docs[best_idx]["chk_id"],
                    "doc_source": selected_docs[best_idx]["newsinfo"]["source"],
                    "doc_date": selected_docs[best_idx]["newsinfo"]["date"],
                    "doc_title": selected_docs[best_idx]["newsinfo"]["title"],
                    "chk_content": selected_docs[best_idx]['content'],
                    "best_ratio": citation['best_ratio'],
                    "highlight": citation['highlighted_start_end'],
                }
                group_list.append(group_item)
                quote_list.append({
                    "doc_list": group_list,
                    "chk_content": group_list[0]["chk_content"],
                    "highlight": group_list[0]["highlight"],
                })
                best_indices += 1
                final_response.append(f"{citation['title']}{[best_indices]}\n\n")
                # final_response.append(f"{citation['title']}\n")
                # final_response.append(f"\n{citation['content']}{[best_indices]}\n\n")

                is_content_exists.append(best_idx)
        data = {'result': ''.join(final_response), 'quote_list': quote_list, 'summary': response['summary']}
        return data


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
