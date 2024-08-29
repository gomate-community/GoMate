import pickle
from typing import List

import jieba
class MatchCitation:
    def __init__(self):
        self.stopwords = ["的"]

    def cut(self, para: str):
        return para.split("。")

    def remove_stopwords(self, query: str):
        for word in self.stopwords:
            query = query.replace(word, " ")
        return query

    # def ground_response(
    #         self,
    #         response: str,
    #         evidences: List[str],
    #         selected_idx: List[int],
    #         markdown: bool = True,
    #         show_code=True,
    #         selected_docs=List[dict]
    # ):
    #     """
    #      # selected_docs:[ {"file_name": 'source', "content":'xxxx' , "chk_id": 1,"doc_id": '11', "newsinfo": {'title':'xxx','source':'xxx','date':'2024-08-25'}}]
    #         # if best_ratio > threshold:
    #         #     final_response.append(f"{sentence}[{best_idx+1}]。")
    #         #     if show_code:
    #         #         final_response.append(f"\n```\n{best_match}。\n```\n")
    #         # else:
    #         #     final_response.append(f"{sentence}。")
    #     """
    #     # selected_idx[-1]=selected_idx[-1]-1
    #     print(selected_idx)
    #     sentences = self.cut(response)
    #     final_response = []
    #     print("\n==================response===================\n",response)
    #     print("\n==================evidences===================\n",evidences)
    #     print("\n==================selected_idx===================\n",selected_idx)
    #     selected_idx=[i-1 for i in selected_idx]
    #     print("\n==================selected_idx===================\n", selected_idx)
    #     print("\n==================len(evidences)===================\n", len(evidences))
    #     print("\n==================len(selected_docs)===================\n", len(selected_docs))
    #
    #     for sentence in sentences:
    #         if not sentence.strip():
    #             continue
    #
    #         sentence_seg_cut = set(jieba.lcut(self.remove_stopwords(sentence)))
    #         sentence_seg_cut_length = len(sentence_seg_cut)
    #
    #         best_match = None
    #         best_ratio = 0
    #         best_idx = None
    #         best_i=None
    #         for i,idx in enumerate(selected_idx):
    #             evidence = evidences[i]
    #             evidence_sentences = self.cut(evidence)
    #
    #             for evidence_sentence in evidence_sentences:
    #                 evidence_seg_cut = set(jieba.lcut(self.remove_stopwords(evidence_sentence)))
    #                 overlap = sentence_seg_cut.intersection(evidence_seg_cut)
    #                 ratio = len(overlap) / sentence_seg_cut_length
    #
    #                 if ratio > best_ratio:
    #                     best_ratio = ratio
    #                     best_match = evidence_sentence
    #                     best_idx = idx + 1
    #                     best_i=i
    #         threshold = 0.7 if len(sentence) > 20 else 0.6
    #
    #
    #         if best_ratio > threshold:
    #             final_response.append(f"{sentence}[{best_idx+1}]。")
    #             if show_code:
    #                 doc_info = selected_docs[best_i]
    #                 newsinfo = doc_info.get('newsinfo', {})
    #                 source = newsinfo.get('source', '')
    #                 date = newsinfo.get('date', '')
    #                 title = newsinfo.get('title', '')
    #
    #                 info_string = f"来源: {source}, 日期: {date}, 标题: {title}"
    #                 final_response.append(f"\n```python\n{info_string}\n\n{best_match}。\n```\n")
    #         else:
    #             final_response.append(f"{sentence}。")
    #
    #     return ''.join(final_response)

    # def highlight_matching_segments(self, sentence, text):
    #     # 将句子和文本分词
    #     sentence_words = jieba.lcut(sentence)
    #     text_words = jieba.lcut(text)
    #
    #     # 找出匹配的词
    #     matching_words = set(sentence_words) & set(text_words)
    #
    #     # 高亮匹配的词
    #     highlighted_words = []
    #     for word in text_words:
    #         if word in matching_words:
    #             highlighted_words.append(f"\033[1;33m{word}\033[0m")  # 黄色高亮
    #         else:
    #             highlighted_words.append(word)
    #
    #     return ''.join(highlighted_words)
    # def ground_response(
    #         self,
    #         response: str,
    #         evidences: List[str],
    #         selected_idx: List[int],
    #         markdown: bool = True,
    #         show_code=True,
    #         selected_docs=List[dict]
    # ):
    #     sentences = self.cut(response)
    #     final_response = []
    #     selected_idx = [i - 1 for i in selected_idx]
    #
    #     for sentence in sentences:
    #         if not sentence.strip():
    #             continue
    #
    #         sentence_seg_cut = set(jieba.lcut(self.remove_stopwords(sentence)))
    #         sentence_seg_cut_length = len(sentence_seg_cut)
    #
    #         best_match = None
    #         best_ratio = 0
    #         best_idx = None
    #         best_i = None
    #
    #         for i, idx in enumerate(selected_idx):
    #             evidence = evidences[i]
    #             evidence_sentences = self.cut(evidence)
    #
    #             for j, evidence_sentence in enumerate(evidence_sentences):
    #                 evidence_seg_cut = set(jieba.lcut(self.remove_stopwords(evidence_sentence)))
    #                 overlap = sentence_seg_cut.intersection(evidence_seg_cut)
    #                 ratio = len(overlap) / sentence_seg_cut_length
    #
    #                 if ratio > best_ratio:
    #                     best_ratio = ratio
    #                     best_match = evidence_sentence
    #                     best_idx = idx + 1
    #                     best_i = i
    #                     best_j = j
    #
    #         threshold = 0.7 if len(sentence) > 20 else 0.6
    #
    #         if best_ratio > threshold:
    #             final_response.append(f"{sentence}[{best_idx + 1}]。")
    #             if show_code:
    #                 doc_info = selected_docs[best_i]
    #                 newsinfo = doc_info.get('newsinfo', {})
    #                 source = newsinfo.get('source', '')
    #                 date = newsinfo.get('date', '')
    #                 title = newsinfo.get('title', '')
    #
    #                 info_string = f"来源: {source}, 日期: {date}, 标题: {title}"
    #
    #                 # 优化1: 如果best_match长度小于80，拼接上下文
    #                 evidence_sentences = self.cut(evidences[best_i])
    #                 if len(best_match) < 80:
    #                     start = max(0, best_j - 1)
    #                     end = min(len(evidence_sentences), best_j + 2)
    #                     best_match = ' '.join(evidence_sentences[start:end])
    #
    #                 # 优化2: 高亮匹配片段
    #                 highlighted_match = self.highlight_matching_segments(sentence, best_match)
    #
    #                 # 优化3: 灰色显示info_string
    #                 final_response.append(f"\n```python\n\033[90m{info_string}\033[0m\n\n{highlighted_match}。\n```\n")
    #         else:
    #             final_response.append(f"{sentence}。")
    #
    #     return ''.join(final_response)

    # def ground_response(
    #         self,
    #         response: str,
    #         evidences: List[str],
    #         selected_idx: List[int],
    #         markdown: bool = True,
    #         show_code=True,
    #         selected_docs=List[dict]
    # ):
    #     sentences = self.cut(response)
    #     final_response = []
    #     selected_idx = [i - 1 for i in selected_idx]
    #
    #     for sentence in sentences:
    #         if not sentence.strip():
    #             continue
    #
    #         sentence_seg_cut = set(jieba.lcut(self.remove_stopwords(sentence)))
    #         sentence_seg_cut_length = len(sentence_seg_cut)
    #
    #         best_match = None
    #         best_ratio = 0
    #         best_idx = None
    #         best_i = None
    #
    #         for i, idx in enumerate(selected_idx):
    #             evidence = evidences[i]
    #             evidence_sentences = self.cut(evidence)
    #
    #             for j, evidence_sentence in enumerate(evidence_sentences):
    #                 evidence_seg_cut = set(jieba.lcut(self.remove_stopwords(evidence_sentence)))
    #                 overlap = sentence_seg_cut.intersection(evidence_seg_cut)
    #                 ratio = len(overlap) / sentence_seg_cut_length
    #
    #                 if ratio > best_ratio:
    #                     best_ratio = ratio
    #                     best_match = evidence_sentence
    #                     best_idx = idx + 1
    #                     best_i = i
    #                     best_j = j
    #
    #         threshold = 0.7 if len(sentence) > 20 else 0.6
    #
    #         if best_ratio > threshold:
    #             final_response.append(f"{sentence}[{best_idx + 1}]。")
    #             if show_code:
    #                 doc_info = selected_docs[best_i]
    #                 newsinfo = doc_info.get('newsinfo', {})
    #                 source = newsinfo.get('source', '')
    #                 date = newsinfo.get('date', '')
    #                 title = newsinfo.get('title', '')
    #
    #                 info_string = f"来源: {source}, 日期: {date}, 标题: {title}"
    #
    #                 # 优化1: 如果best_match长度小于80，拼接上下文
    #                 evidence_sentences = self.cut(evidences[best_i])
    #                 if len(best_match) < 80:
    #                     start = max(0, best_j - 1)
    #                     end = min(len(evidence_sentences), best_j + 2)
    #                     best_match = ' '.join(evidence_sentences[start:end])
    #
    #                 # 优化2: 使用下划线标记匹配片段
    #                 highlighted_match = self.underline_matching_segments(sentence, best_match)
    #
    #                 # 优化3: 使用 markdown 语法为 info_string 添加灰色
    #                 final_response.append(f"\n```python\n*{info_string}*\n\n{highlighted_match}。\n```\n")
    #         else:
    #             final_response.append(f"{sentence}。")
    #
    #     return ''.join(final_response)
    #
    # def underline_matching_segments(self, sentence, text):
    #     # 将句子和文本分词
    #     sentence_words = jieba.lcut(sentence)
    #     text_words = jieba.lcut(text)
    #
    #     # 找出匹配的词
    #     matching_words = set(sentence_words) & set(text_words)
    #
    #     # 为匹配的词添加下划线
    #     underlined_words = []
    #     for word in text_words:
    #         if word in matching_words:
    #             underlined_words.append(f"__{word}__")  # 使用双下划线标记
    #         else:
    #             underlined_words.append(word)
    #
    #     return ''.join(underlined_words)

    def ground_response(
            self,
            response: str,
            evidences: List[str],
            selected_idx: List[int],
            markdown: bool = True,
            show_code=True,
            selected_docs=List[dict]
    ):
        sentences = self.cut(response)
        final_response = []
        selected_idx = [i - 1 for i in selected_idx]

        for sentence in sentences:
            if not sentence.strip():
                continue

            sentence_seg_cut = set(jieba.lcut(self.remove_stopwords(sentence)))
            sentence_seg_cut_length = len(sentence_seg_cut)

            best_match = None
            best_ratio = 0
            best_idx = None
            best_i = None

            for i, idx in enumerate(selected_idx):
                evidence = evidences[i]
                evidence_sentences = self.cut(evidence)

                for j, evidence_sentence in enumerate(evidence_sentences):
                    evidence_seg_cut = set(jieba.lcut(self.remove_stopwords(evidence_sentence)))
                    overlap = sentence_seg_cut.intersection(evidence_seg_cut)
                    ratio = len(overlap) / sentence_seg_cut_length

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match = evidence_sentence
                        best_idx = idx + 1
                        best_i = i
                        best_j = j

            threshold = 0.7 if len(sentence) > 20 else 0.6

            if best_ratio > threshold:
                final_response.append(f"{sentence}[{best_idx + 1}]。")
                if show_code:
                    doc_info = selected_docs[best_i]
                    newsinfo = doc_info.get('newsinfo', {})
                    source = newsinfo.get('source', '')
                    date = newsinfo.get('date', '')
                    title = newsinfo.get('title', '')

                    info_string = f"来源: {source}, 日期: {date}, 标题: {title}"

                    #  如果best_match长度小于80，拼接上下文
                    evidence_sentences = self.cut(evidences[best_i])
                    print(best_match)
                    print(len(best_match))
                    if best_match and len(best_match) < 80:
                        start = max(0, best_j - 1)
                        end = min(len(evidence_sentences), best_j + 2)
                        best_match = ' '.join(evidence_sentences[start:end])
                        print(f"Extended Best Match: {best_match}")

                    # 优化2: 使用HTML标签标记匹配片段
                    highlighted_match = self.highlight_common_substrings(sentence, best_match)

                    # 优化3: 使用HTML标签为info_string添加灰色
                    final_response.append(
                        f"\n> <span style='color:gray'>{info_string}</span>\n>\n> {highlighted_match}。\n\n")
            else:
                final_response.append(f"{sentence}。")

        return ''.join(final_response)

    # def highlight_common_substrings(self, str1, str2, min_length=6):
    #     def find_common_substrings(s1, s2, min_len):
    #         m, n = len(s1), len(s2)
    #         dp = [[0] * (n + 1) for _ in range(m + 1)]
    #         substrings = []
    #
    #         for i in range(1, m + 1):
    #             for j in range(1, n + 1):
    #                 if s1[i - 1] == s2[j - 1]:
    #                     dp[i][j] = dp[i - 1][j - 1] + 1
    #                     if dp[i][j] >= min_len:
    #                         substrings.append((i - dp[i][j], i, j - dp[i][j], j))
    #                 else:
    #                     dp[i][j] = 0
    #
    #         return sorted(substrings, key=lambda x: x[2], reverse=True)  # 按str2中的起始位置排序
    #
    #     common_substrings = find_common_substrings(str1, str2, min_length)
    #
    #     # 标记需要高亮的部分
    #     marked_positions = [0] * len(str2)
    #     for _, _, start2, end2 in common_substrings:
    #         for i in range(start2, end2):
    #             marked_positions[i] = 1
    #
    #     # 构建带有高亮标记的字符串
    #     result = []
    #     in_mark = False
    #     for i, char in enumerate(str2):
    #         if marked_positions[i] and not in_mark:
    #             result.append("<mark>")
    #             in_mark = True
    #         elif not marked_positions[i] and in_mark:
    #             result.append("</mark>")
    #             in_mark = False
    #         result.append(char)
    #
    #     if in_mark:
    #         result.append("</mark>")
    #
    #     return ''.join(result)
    def highlight_common_substrings(self, str1, str2, min_length=6):
        def find_common_substrings(s1, s2, min_len):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            substrings = []

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i - 1] == s2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                        if dp[i][j] >= min_len:
                            substrings.append((i - dp[i][j], i, j - dp[i][j], j))
                    else:
                        dp[i][j] = 0

            return sorted(substrings, key=lambda x: x[2], reverse=True)  # 按str2中的起始位置排序

        common_substrings = find_common_substrings(str1, str2, min_length)

        # 标记需要高亮的部分
        marked_positions = [0] * len(str2)
        for _, _, start2, end2 in common_substrings:
            for i in range(start2, end2):
                marked_positions[i] = 1

        # 构建带有蓝色高亮标记的字符串
        result = []
        in_mark = False
        for i, char in enumerate(str2):
            if marked_positions[i] and not in_mark:
                result.append("<span style='color:blue;text-decoration:underline'>")
                in_mark = True
            elif not marked_positions[i] and in_mark:
                result.append("</span>")
                in_mark = False
            result.append(char)

        if in_mark:
            result.append("</span>")

        return ''.join(result)
if __name__ == '__main__':
    mc = MatchCitation()

    result = mc.ground_response(
        response="巨齿鲨2是一部科幻冒险电影，由本·维特利执导，杰森·斯坦森、吴京、蔡书雅和克利夫·柯蒂斯主演。电影讲述了海洋霸主巨齿鲨，今夏再掀狂澜！乔纳斯·泰勒（杰森·斯坦森饰）与科学家张九溟（吴京饰）双雄联手，进入海底7000米深渊执行探索任务。他们意外遭遇史前巨兽海洋霸主巨齿鲨群的攻击，还将对战凶猛危险的远古怪兽群。惊心动魄的深渊冒险，巨燃巨爽的深海大战一触即发。",
        evidences=[
            "海洋霸主巨齿鲨，今夏再掀狂澜！乔纳斯·泰勒（杰森·斯坦森 饰）与科学家张九溟（吴京 饰）双雄联手，进入海底7000米深渊执行探索任务。他们意外遭遇史前巨兽海洋霸主巨齿鲨群的攻击，还将对战凶猛危险的远古怪兽群。惊心动魄的深渊冒险，巨燃巨爽的深海大战一触即发",
            "本·维特利 编剧：乔·霍贝尔埃里希·霍贝尔迪恩·乔格瑞斯 国家地区：中国 | 美国 发行公司：上海华人影业有限公司五洲电影发行有限公司中国电影股份有限公司北京电影发行分公司 出品公司：上海华人影业有限公司华纳兄弟影片公司北京登峰国际文化传播有限公司 更多片名：巨齿鲨2 剧情：海洋霸主巨齿鲨，今夏再掀狂澜！乔纳斯·泰勒（杰森·斯坦森 饰）与科学家张九溟（吴京 饰）双雄联手，进入海底7000米深渊执行探索任务。他们意外遭遇史前巨兽海洋霸主巨齿鲨群的攻击，还将对战凶猛危险的远古怪兽群。惊心动魄的深渊冒险，巨燃巨爽的深海大战一触即发……"
        ],
        selected_idx=[0, 1],
        markdown=True,
        show_code=True
    )

    print(result)
