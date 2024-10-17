import json
from typing import List

import jieba
import loguru

from gomate.modules.document.utils import PROJECT_BASE


class MatchCitation:
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

    def highlight_common_substrings(self, sentence, evidence_sentence, evidence, min_length=6):
        evidence_sentences = self.cut(evidence)
        current_sentence_index = next(i for i, s in enumerate(evidence_sentences) if evidence_sentence == s)
        highlighted_text = evidence_sentences[current_sentence_index]
        start_evidence = evidence.index(highlighted_text)
        end_evidence = start_evidence + len(highlighted_text)
        return [[start_evidence, end_evidence - 1]]

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
        """
        """
        # Create JSON object
        json_data = {
            "question": question,
            "response": response,
            "evidences": evidences,
            "selected_idx": selected_idx,
            "selected_docs": selected_docs
        }
        # Save to JSON file
        output_file = "citation_match.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        loguru.logger.info(f"Parameters saved to {output_file}")
        sentences = self.cut(response)

        contents = [{"content": sentence} for sentence in sentences]
        for cit_idx, citation in enumerate(contents):
            citation['citation_content'] = []
            citation['best_idx'] = []
            citation['best_ratio'] = []
            citation['highlighted_start_end'] = []
            sentence = citation['content']
            # print("===================sentence", sentence)
            # 答案内容进行分词
            sentence_seg_cut = set(jieba.lcut(self.remove_stopwords(sentence)))
            sentence_seg_cut_length = len(sentence_seg_cut)

            threshold = 0.5
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
                        highlighted_start_end = self.highlight_common_substrings(sentence, evidence_sentence,
                                                                                 doc['content'])
                        if best_idx not in citation['best_idx']:
                            citation['citation_content'].append(doc['content'])
                            citation['best_idx'].append(best_idx)
                            citation['best_ratio'].append(best_ratio)
                            citation['highlighted_start_end'].append(highlighted_start_end)
        print(contents)

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

        is_group_exists=[]
        for citation_idx, citation in enumerate(contents):
            final_response.append(f"{citation['content']}")

            best_idxes = citation['best_idx']
            if len(best_idxes) > 0:
                # print(citation)
                # print(best_idxes)
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

                    merged_group_list = []
                    reference = group_list[0]
                    reference_tokens = set(jieba.lcut(self.remove_stopwords(reference['chk_content'])))
                    merged_group = [reference]
                    for item in group_list[1:]:
                        item_tokens = set(jieba.lcut(self.remove_stopwords(item['chk_content'])))
                        if len(reference_tokens.intersection(item_tokens)) > 5:
                            merged_group.append(item)
                        else:
                            merged_group_list.append([item])
                            # merged_group = [item]
                    if merged_group:
                        merged_group_list.append(merged_group)
                    for group in merged_group_list:
                        group_data={
                                "doc_list": group,
                                "chk_content": group[0]["chk_content"],
                                "highlight": group[0]["highlight"],
                            }
                        doc_id_list=[doc['doc_id'] for doc in group_data['doc_list']]
                        # print(doc_id_list)
                        if doc_id_list not in is_group_exists:
                            quote_list.append(group_data)
                            best_indices += 1
                            final_response.append(f"{[best_indices]}")
                            is_group_exists.append(doc_id_list)
                        else:
                            # print("已存在")
                            final_response.append(f"{[is_group_exists.index(doc_id_list)+1]}")

        data = {'result': ''.join(final_response), 'quote_list': quote_list, 'summary': ''}
        # Save to JSON file
        json_data['result'] = ''.join(final_response)
        json_data['quote_list'] = quote_list
        output_file = "citation_match_res.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

        loguru.logger.info(f"Parameters saved to {output_file}")
        print(json_data)
        return data


if __name__ == '__main__':
    mc = MatchCitation()

    with open(f'{PROJECT_BASE}/data/docs/citations_samples/sample18.json', 'r', encoding='utf-8') as f:
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
