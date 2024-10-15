import json
from typing import List
from  gomate.modules.document.utils import PROJECT_BASE
import jieba
import loguru


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

    def ground_response(
            self,
            question:str,
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

        # Log using loguru
        # loguru.logger.info(f"Response: {response}")
        # loguru.logger.info(f"Evidences: {evidences}")
        # loguru.logger.info(f"Selected indices: {selected_idx}")
        # loguru.logger.info(f"Selected documents: {selected_docs}")

        # Save to JSON file
        output_file = "citation.json"
        with open("citation.json", 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

        loguru.logger.info(f"Parameters saved to {output_file}")

        print(response)
        sentences = self.cut(response)
        print(sentences)
        selected_idx = [i - 1 for i in selected_idx]

        quote_list = []
        final_response = []
        quote_index_map = {}  # To keep track of existing quotes
        best_idx = 0

        for sentence in sentences:
            print("===================sentence", sentence)
            if not sentence.strip():
                # continue
                final_response.append(sentence)
            else:
                sentence_seg_cut = set(jieba.lcut(self.remove_stopwords(sentence)))
                sentence_seg_cut_length = len(sentence_seg_cut)
                threshold = 0.6
                final_response.append(f"{sentence}")
                group_list = []
                for i, idx in enumerate(selected_idx):
                    evidence = evidences[i]
                    evidence_sentences = self.cut(evidence)
                    for j, evidence_sentence in enumerate(evidence_sentences):
                        evidence_seg_cut = set(jieba.lcut(self.remove_stopwords(evidence_sentence)))
                        overlap = sentence_seg_cut.intersection(evidence_seg_cut)
                        ratio = len(overlap) / sentence_seg_cut_length
                        if ratio > threshold:
                            best_ratio = ratio
                            highlighted_start_end = self.highlight_common_substrings(sentence, evidence_sentence, evidence)
                            group_item = {
                                "doc_id": selected_docs[i]["doc_id"],
                                "chk_id": selected_docs[i]["chk_id"],
                                "doc_source": selected_docs[i]["newsinfo"]["source"],
                                "doc_date": selected_docs[i]["newsinfo"]["date"],
                                "doc_title": selected_docs[i]["newsinfo"]["title"],
                                "chk_content": evidence,
                                "best_ratio": best_ratio,
                                "highlight": highlighted_start_end,
                            }
                            group_list.append(group_item)

                if group_list:
                    # Create a unique key for the group_list based on its content
                    group_key = tuple(sorted((item["doc_id"], item["chk_id"]) for item in group_list))

                    if group_key in quote_index_map:
                        # If this group already exists, use its index
                        existing_idx = quote_index_map[group_key]
                        final_response.append(f"[{existing_idx}]")
                    else:
                        # If this is a new group, add it to quote_list and update the index
                        best_idx += 1
                        quote_index_map[group_key] = best_idx
                        quote_list.append({
                            "doc_list": group_list,
                            "chk_content": group_list[0]["chk_content"],
                            "highlight": group_list[0]["highlight"],
                        })
                        final_response.append(f"[{best_idx}]")

            # final_response.append("。")
            # final_response.append("\n")
        # print(''.join(final_response))
        data = {'result': ''.join(final_response), 'quote_list': quote_list,'summary':''}
        return data

    def highlight_common_substrings(self, sentence, evidence_sentence, evidence, min_length=6):
        evidence_sentences = self.cut(evidence)
        current_sentence_index = next(i for i, s in enumerate(evidence_sentences) if evidence_sentence == s)
        highlighted_text = evidence_sentences[current_sentence_index]
        start_evidence = evidence.index(highlighted_text)
        end_evidence = start_evidence + len(highlighted_text)
        return [[start_evidence, end_evidence-1]]


if __name__ == '__main__':
    mc = MatchCitation()

    with open(f'{PROJECT_BASE}/data/docs/citations_samples/sample1.json','r',encoding='utf-8') as f:
        input_data =json.load(f)
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
