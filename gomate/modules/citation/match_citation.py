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

    def ground_response(
            self,
            response: str,
            evidences: List[str],
            selected_idx: List[int],
            markdown: bool = True,
            show_code=False,
            selected_docs=List[dict]
    ):
        """
         data: {
        result: '中国共产党是中国工人阶级的先锋队，同时是中国人民和中华民族的先锋队，是中国特色社会主义事业的领导核心，代表中国先进生产力的发展要求，代表中国先进文化的前进方向，代表中国最广大人民的根本利益。[1][2][3][4][5]党的最高理想和最终目标是实现共产主义。[3][4][5]中国共产党以马克思列宁主义、毛泽东思想、邓小平理论、“三个代表”重要思想、科学发展观、习近平新时代中国特色社会主义思想作为自己的行动指南。[3][4][5]党必须适应形势的发展和情况的变化，完善领导体制，改进领导方式，增强执政能力。[5]共产党员必须同党外群众亲密合作，共同为建设中国特色社会主义而奋斗。[3][4][5]',
        quote_list: [
            // 文内第一个引用
            {
                "doc_id": 90564, // 文件id
                "chk_id":3， // 切片索引（从0开始）
                // 非文内溯源知识集合无需返回
                "doc_source": "新闻来源",
                // 新闻时间， 非文内溯源知识集合无需返回
                "doc_date": "2021-10-19",
                // 非文内溯源知识集合无需返回
                "doc_title": "新闻标题",
                // 非文内溯源知识集合无需返回
                "chk_content":"切片文本",
                // 非文内溯源知识集合无需返回
                // 高亮文本在chk_content中的起始索引（从0开始）
                "highlight":[10， 21] ,
            },
            // 文内第二个引用
            {
                "doc_id": 90564, // 文件id
                "chk_id":4， // 切片索引（从0开始）
                // 非文内溯源知识集合无需返回
                "doc_source": "新闻来源",
                // 新闻时间， 非文内溯源知识集合无需返回
                "doc_date": "2021-10-19",
                // 非文内溯源知识集合无需返回
                "doc_title": "新闻标题",
                // 非文内溯源知识集合无需返回
                "chk_content":"切片文本",
                // 非文内溯源知识集合无需返回
                // 高亮文本在chk_content中的起始索引（从0开始）
                "highlight":[10， 21] ,
            },
            // ....
        ]
      },
        """
        print(selected_docs)
        sentences = self.cut(response)
        final_response = []
        selected_idx = [i - 1 for i in selected_idx]

        quote_list = []
        best_idx=0
        for sentence in sentences:
            if not sentence.strip():
                continue

            sentence_seg_cut = set(jieba.lcut(self.remove_stopwords(sentence)))
            sentence_seg_cut_length = len(sentence_seg_cut)

            threshold = 0.7 if len(sentence) > 20 else 0.6
            final_response.append(f"{sentence}")

            for i, idx in enumerate(selected_idx):
                evidence = evidences[i]
                evidence_sentences = self.cut(evidence)
                for j, evidence_sentence in enumerate(evidence_sentences):
                    evidence_seg_cut = set(jieba.lcut(self.remove_stopwords(evidence_sentence)))
                    overlap = sentence_seg_cut.intersection(evidence_seg_cut)
                    ratio = len(overlap) / sentence_seg_cut_length
                    if ratio > threshold:
                        best_ratio = ratio
                        # best_match = evidence_sentence
                        # best_idx = idx + 1
                        final_response.append(f"[{best_idx + 1}]")
                        # todo:返回高亮位置
                        highlighted_start_end = self.highlight_common_substrings(sentence, evidence_sentence,evidence)
                        quote_list.append(
                            {
                                "doc_id": selected_docs[i]["doc_id"],  # 文件id
                                "chk_id": selected_docs[i]["chk_id"],  # 切片索引（从0开始）
                                # 非文内溯源知识集合无需返回
                                "doc_source": selected_docs[i]["newsinfo"]["source"],
                                # 新闻时间, 非文内溯源知识集合无需返回
                                "doc_date": selected_docs[i]["newsinfo"]["date"],
                                # 非文内溯源知识集合无需返回
                                "doc_title": selected_docs[i]["newsinfo"]["title"],
                                # 非文内溯源知识集合无需返回
                                "chk_content": evidence,
                                "best_ratio": best_ratio,
                                # 非文内溯源知识集合无需返回
                                # 高亮文本在chk_content中的起始索引（从0开始） [10， 21]
                                "highlight": highlighted_start_end,
                            }
                        )
                        best_idx+=1
            final_response.append("。")
        data={'result':''.join(final_response),'quote_list':quote_list}
        return data

    # def highlight_common_substrings(self, sentence, evidence_sentence, evidence, min_length=6):
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
    #         return sorted(substrings, key=lambda x: x[1] - x[0], reverse=True)  # Sort by length, longest first
    #
    #     common_substrings = find_common_substrings(sentence, evidence_sentence, min_length)
    #
    #     if not common_substrings:
    #         return []
    #
    #     # Get the longest common substring
    #     _, _, start_evidence_sentence, end_evidence_sentence = common_substrings[0]
    #
    #     # Find the position of evidence_sentence in evidence
    #     evidence_sentence_start = evidence.index(evidence_sentence)
    #
    #     # Calculate the actual start and end positions in evidence
    #     start_evidence = evidence_sentence_start + start_evidence_sentence
    #     end_evidence = evidence_sentence_start + end_evidence_sentence
    #
    #     return [[start_evidence, end_evidence]]
    def highlight_common_substrings(self, sentence, evidence_sentence, evidence, min_length=6):
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

            return sorted(substrings, key=lambda x: x[1] - x[0], reverse=True)  # Sort by length, longest first

        common_substrings = find_common_substrings(sentence, evidence_sentence, min_length)

        if not common_substrings:
            return []

        # Get the longest common substring
        _, _, start_evidence_sentence, end_evidence_sentence = common_substrings[0]

        # Find the position of evidence_sentence in evidence
        # evidence_sentence_start = evidence.index(evidence_sentence)

        # Split evidence into sentences
        evidence_sentences = self.cut(evidence)

        # Find the index of the current sentence
        current_sentence_index = next(i for i, s in enumerate(evidence_sentences) if evidence_sentence == s)

        # Get surrounding sentences
        start_sentence_index = max(0, current_sentence_index - 1)
        end_sentence_index = min(len(evidence_sentences) - 1, current_sentence_index + 1)

        # Join the sentences
        highlighted_text = '。'.join(evidence_sentences[start_sentence_index:end_sentence_index + 1])
        print("highlighted_text====>",highlighted_text)
        print("evidence",evidence)
        # Calculate the new start and end positions
        start_evidence = evidence.index(highlighted_text)
        end_evidence = start_evidence + len(highlighted_text)

        return [[start_evidence, end_evidence]]

    # def highlight_common_substrings(self, sentence, evidence_sentence, evidence, min_length=6):
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
    #         return sorted(substrings, key=lambda x: x[1] - x[0], reverse=True)  # Sort by length, longest first
    #
    #     common_substrings = find_common_substrings(sentence, evidence_sentence, min_length)
    #
    #     if not common_substrings:
    #         return []
    #
    #     # Get the longest common substring
    #     _, _, start_evidence_sentence, end_evidence_sentence = common_substrings[0]
    #
    #     # Split evidence into sentences
    #     evidence_sentences = self.cut(evidence)
    #
    #     # Find the index of the current sentence
    #     current_sentence_index = next(i for i, s in enumerate(evidence_sentences) if evidence_sentence == s)
    #
    #     # Get surrounding sentences
    #     start_sentence_index = max(0, current_sentence_index - 1)
    #     end_sentence_index = min(len(evidence_sentences) - 1, current_sentence_index + 1)
    #
    #     # Calculate start_evidence and end_evidence
    #     start_evidence = sum(len(s) for s in evidence_sentences[:start_sentence_index])
    #     end_evidence = sum(len(s) for s in evidence_sentences[:end_sentence_index + 1])
    #
    #     # Adjust start_evidence and end_evidence to include any separators
    #     while start_evidence > 0 and evidence[start_evidence - 1] in '。':
    #         start_evidence -= 1
    #     while end_evidence < len(evidence) and evidence[end_evidence] in '。':
    #         end_evidence += 1
    #
    #     return [[start_evidence, end_evidence - 1]]

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
