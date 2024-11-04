import json
import re
from typing import List

import jieba
import loguru

from gomate.modules.document.utils import PROJECT_BASE


class SourceCitation:
    def __init__(self):
        self.stopwords = ["的"]

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

    def extract_content(self, text):
        """
        从文本中提取summary和所有的title-content对

        参数:
            text: 包含summary和多个title-content对的文本

        返回:
            tuple: (summary, list of dict)
        """
        import re

        # 提取summary部分
        summary_pattern = r'"summary"\s*:\s*"([^"]+)"'
        summary_match = re.search(summary_pattern, text)
        summary = summary_match.group(1) if summary_match else ""

        # 提取所有的title-content对
        pattern = r'{[^{]*?"title"\s*:\s*"([^"]+)"[^{]*?"content"\s*:\s*"([^"]+)"[^}]*?}'
        matches = re.finditer(pattern, text)

        items = []
        for match in matches:
            item = {
                "title": match.group(1),
                "content": match.group(2)
            }
            items.append(item)
        result = {
            "summary": summary,
            "contents": items
        }
        return result

    def load_response_json(self, response):
        cleaned_response = re.sub(r'^.*?```json\n|```$', '', response, flags=re.DOTALL)
        print("cleaned_response",cleaned_response)
        try:
            data = json.loads(cleaned_response)
        except:
            data = self.extract_content(cleaned_response)
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

    def merge_groups(self, groups):
        """
        Merge groups based on their highlighted content and chk_content

        Args:
            groups: List of lists containing document dictionaries
        Returns:
            List of merged groups
        """
        if not groups:
            return []

        # Helper function to get highlighted content
        def get_highlighted_content(doc):
            if not doc.get("highlight") or not doc.get("chk_content"):
                return ""
            highlights = []
            for start, end in doc["highlight"]:
                highlights.append(doc["chk_content"][start:end])
            return "".join(highlights)

        # Initialize merged groups with the first group
        merged_groups = [groups[0]]
        highlighted_contents = [get_highlighted_content(groups[0][0])]

        # Iterate through remaining groups
        for current_group in groups[1:]:
            current_highlight = get_highlighted_content(current_group[0])

            # Check if current group should be merged with any existing group
            merged = False
            for i, (existing_group, existing_highlight) in enumerate(zip(merged_groups, highlighted_contents)):
                if (current_highlight == existing_highlight and
                        current_group[0]["chk_content"] == existing_group[0]["chk_content"]):
                    # Merge groups
                    merged_groups[i].extend(current_group)
                    merged = True
                    break

            # If no merge occurred, add as new group
            if not merged:
                merged_groups.append(current_group)
                highlighted_contents.append(current_highlight)

        return merged_groups

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
            "selected_docs": selected_docs,
        }
        raw_response = response
        try:
            output_file = "/home/yanqiang/code/citation.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
        except:
            output_file = "citation.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
        response = self.load_response_json(response)
        contents = [content for content in response['contents'] if 'title' in content and 'content' in content]

        is_citation_used=[]
        for cit_idx, citation in enumerate(contents):
            citation['citation_content'] = []
            citation['best_idx'] = []
            citation['best_ratio'] = []
            citation['highlighted_start_end'] = []
            # 生成的答案内容：citation['title']，citation['content']
            sentence = citation['title'] + citation['content']
            # 答案内容进行分词
            sentence_seg_cut = set(jieba.lcut(self.remove_stopwords(sentence)))
            sentence_seg_cut_length = len(sentence_seg_cut)

            threshold = 0.25
            # 检索内容
            for doc_idx, doc in enumerate(selected_docs):
                evidence_sentences = self.cut(doc['content'])
                for es_idx, evidence_sentence in enumerate(evidence_sentences):
                    evidence_seg_cut = set(jieba.lcut(self.remove_stopwords(evidence_sentence)))
                    overlap = sentence_seg_cut.intersection(evidence_seg_cut)
                    ratio = len(overlap) / sentence_seg_cut_length
                    if ratio > threshold:
                        best_ratio = ratio
                        best_idx = doc_idx
                        best_sentence = evidence_sentence
                        highlighted_start_end = self.highlight_common_substrings(sentence, evidence_sentence,doc['content'])
                        if best_idx not in citation['best_idx'] :
                            # 跟引用是否被使用过，保障独占式
                            if  (best_idx,highlighted_start_end) not in is_citation_used:
                                citation['citation_content'].append(doc['content'])
                                citation['best_idx'].append(best_idx)
                                citation['best_ratio'].append(best_ratio)
                                citation['highlighted_start_end'].append(highlighted_start_end)
                                is_citation_used.append((best_idx,highlighted_start_end))

        print(len(contents))
        contents = [content for content in contents if content['best_idx']]
        print(len(contents))

        citation_cnt = 0
        is_citation_exists = []
        for citation in contents:
            best_idx = citation['best_idx']
            if best_idx not in is_citation_exists:
                is_citation_exists.append(best_idx)
                citation_cnt += 1

        is_content_exists = []
        final_response = []
        quote_list = []
        best_indices = 0

        for citation_idx, citation in enumerate(contents):
            if citation_cnt > 1:
                citation['title'] = self.convert_to_chinese(str(citation_idx + 1)) + '、' + citation['title']
            citation['title'] = "**" + citation['title'] + "**"
            final_response.append(f"{citation['title']}")

            best_idxes = citation['best_idx']
            print("best_idxes", best_idxes)
            is_doc_id_exists = []
            group_list = []
            # 判断当前一组引用是否被当前段落引用过
            if best_idxes not in is_content_exists:
                for idx, best_idx in enumerate(best_idxes):
                    # 判断当前组是否存在重复文档
                    if selected_docs[best_idx]["doc_id"] not in is_doc_id_exists:
                        group_item = {
                            "doc_id": selected_docs[best_idx]["doc_id"],
                            "chk_id": selected_docs[best_idx]["chk_id"],
                            "doc_source": selected_docs[best_idx]["newsinfo"]["source"],
                            "doc_date": selected_docs[best_idx]["newsinfo"]["date"],
                            "doc_title": selected_docs[best_idx]["newsinfo"]["title"],
                            # "chk_content": selected_docs[best_idx]['content'],
                            "chk_content": citation['citation_content'][idx],
                            "best_ratio": citation['best_ratio'][idx],
                            "highlight": citation['highlighted_start_end'][idx],
                        }
                        group_list.append(group_item)
                        is_doc_id_exists.append(selected_docs[best_idx]["doc_id"])

                # 合并引用
                group_list.sort(key=lambda x: x['best_ratio'], reverse=True)

                # 根据分组内容相似度合并
                merged_group_list = []
                reference = group_list[0]
                reference_tokens = set(jieba.lcut(self.remove_stopwords(reference['chk_content'])))
                merged_group = [reference]
                # print(len(group_list))
                for item in group_list[1:]:
                    item_tokens = set(jieba.lcut(self.remove_stopwords(item['chk_content'])))
                    if len(reference_tokens.intersection(item_tokens)) >= 12:
                        # merged_group.append(item)
                    elif len(set(reference['chk_content']).intersection(set(item['chk_content'])))> 30:
                        # print("***"*20)
                        # print(reference['chk_content'])
                        # print(item['chk_content'])
                        # print(len(reference_tokens.intersection(item_tokens)))
                        # print(len(set(reference['chk_content']).intersection(set(item['chk_content']))))
                        merged_group.append(item)
                    else:
                        merged_group_list.append([item])
                        # merged_group = [item]
                if merged_group:
                    # print(len(merged_group))
                    merged_group_list.append(merged_group)

                # 根据分组是否出现合，完全一致 相似内容
                merged_group_list = self.merge_groups(merged_group_list)
                # print(merged_group_list)

                for group in merged_group_list:
                    quote_list.append({
                        "doc_list": group,
                        "chk_content": group[0]["chk_content"],
                        "highlight": group[0]["highlight"],
                    })

                    best_indices += 1
                    final_response.append(f"{[best_indices]}")

        result = ''.join(final_response)
        if not result.strip():
            # 如果答案为空以及总结内容为空，尝试用原始答案回答
            try:
                if not response['summary']:
                    response['summary'] = raw_response
            except Exception as e:
                result = raw_response
        data = {'result': result, 'quote_list': quote_list, 'summary': response['summary']}

        # Save to JSON file
        json_data['result'] = result
        json_data['quote_list'] = quote_list
        output_file = "citation_res.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        loguru.logger.info(f"Parameters saved to {output_file}")
        # print(data)
        return data


if __name__ == '__main__':
    mc = SourceCitation()

    with open(f'{PROJECT_BASE}/data/docs/citations_samples/完全内容重复3.json', 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    # print(input_data)
    result = mc.ground_response(
        question=input_data["question"],
        response=input_data["response"],
        evidences=input_data["evidences"],
        selected_idx=input_data["selected_idx"],
        markdown=True,
        show_code=True,
        selected_docs=input_data["selected_docs"],
    )

    # print(result)
