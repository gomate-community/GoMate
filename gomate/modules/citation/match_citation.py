import re
from abc import ABC
from typing import List

import jieba


class MatchCitation(ABC):
    def __init__(self):
        self.stopwords = [
            "的"
        ]

    def cut(self, para: str):
        """"""
        pattern = [
            '([。！？\?])([^”’])',  # 单字符断句符
            '(\.{6})([^”’])',  # 英文省略号
            '(\…{2})([^”’])',  # 中文省略号
            '([。！？\?][”’])([^，。！？\?])'
        ]
        for i in pattern:
            para = re.sub(i, r"\1\n\2", para)
        para = para.rstrip()
        return para.split("\n")

    def remove_stopwords(self, query: str):
        for word in self.stopwords:
            query = query.replace(word, " ")
        return query

    def ground_response(self,
                        response: str, evidences: List[str],
                        selected_idx: List[int] = None, markdown: bool = False
                        ) -> List[dict]:
        # {'type': 'default', 'texts': ['xxx', 'xxx']}
        # {'type': 'quote', 'texts': ['1', '2']}
        # if field == 'video':
        #     return [{'type': 'default', 'texts': [response]}]

        # Step 1: cut response into sentences, line break is removed
        # print(response)
        sentences = self.cut(response)
        # print(sentences)
        # get removed line break position
        line_breaks = []
        sentences = [s for s in sentences if s]
        for i in range(len(sentences) - 1):
            current_index = response.index(sentences[i])
            next_sentence_index = response.index(sentences[i + 1])
            dummy_next_sentence_index = current_index + len(sentences[i])
            line_breaks.append(response[dummy_next_sentence_index:next_sentence_index])
        line_breaks.append('')
        final_response = []

        citations = [i + 1 for i in selected_idx]
        paragraph_have_citation = False
        paragraph = ""
        for sentence, line_break in zip(sentences, line_breaks):
            origin_sentence = sentence
            paragraph += origin_sentence
            sentence = self.remove_stopwords(sentence)
            sentence_seg_cut = set(jieba.lcut(sentence))
            sentence_seg_cut_length = len(sentence_seg_cut)
            if sentence_seg_cut_length <= 0:
                continue
            topk_evidences = []

            for evidence, idx in zip(evidences, selected_idx):
                evidence_cuts = self.cut(evidence)
                for j in range(len(evidence_cuts)):
                    evidence_cuts[j] = self.remove_stopwords(evidence_cuts[j])
                    evidence_seg_cut = set(jieba.lcut(evidence_cuts[j]))
                    overlap = sentence_seg_cut.intersection(evidence_seg_cut)
                    topk_evidences.append((len(overlap) / sentence_seg_cut_length, idx))

            topk_evidences.sort(key=lambda x: x[0], reverse=True)

            idx = 0
            sentence_citations = []
            if len(sentence) > 20:
                threshold = 0.4
            else:
                threshold = 0.5

            while (idx < len(topk_evidences)) and (topk_evidences[idx][0] > threshold):
                paragraph_have_citation = True
                sentence_citations.append(topk_evidences[idx][1] + 1)
                if topk_evidences[idx][1] + 1 in citations:
                    citations.remove(topk_evidences[idx][1] + 1)
                idx += 1

            if sentence != sentences[-1] and line_break and line_break[0] == '\n' or sentence == sentences[-1] and len(
                    citations) == 0:
                if not paragraph_have_citation and len(selected_idx) > 0:
                    topk_evidences = []
                    for evidence, idx in zip(evidences, selected_idx):
                        evidence = self.remove_stopwords(evidence)
                        paragraph_seg = set(jieba.lcut(paragraph))
                        evidence_seg = set(jieba.lcut(evidence))
                        overlap = paragraph_seg.intersection(evidence_seg)
                        paragraph_seg_length = len(paragraph_seg)
                        topk_evidences.append((len(overlap) / paragraph_seg_length, idx))
                    topk_evidences.sort(key=lambda x: x[0], reverse=True)
                    if len(paragraph) > 60:
                        threshold = 0.2
                    else:
                        threshold = 0.3
                    if topk_evidences[0][0] > threshold:
                        sentence_citations.append(topk_evidences[0][1] + 1)
                        if topk_evidences[0][1] + 1 in citations:
                            citations.remove(topk_evidences[0][1] + 1)
                paragraph_have_citation = False
                paragraph = ""

            # Add citation to response, need to consider the punctuation and line break
            if origin_sentence[-1] not in [':', '：'] and len(origin_sentence) > 10 and len(sentence_citations) > 0:
                sentence_citations = list(set(sentence_citations))
                if origin_sentence[-1] in ['。', '，', '！', '？', '.', ',', '!', '?', ':', '：']:
                    if markdown:
                        final_response.append(
                            origin_sentence[:-1] + ''.join([f'[{c}]' for c in sentence_citations]) + origin_sentence[
                                -1])
                    else:
                        final_response.append({'type': 'default', 'texts': [origin_sentence[:-1]]})
                        final_response.append({'type': 'quote', 'texts': [str(c) for c in sentence_citations]})
                        final_response.append({'type': 'default', 'texts': [origin_sentence[-1]]})
                else:
                    if markdown:
                        final_response.append(origin_sentence + ''.join([f'[{c}]' for c in sentence_citations]))
                    else:
                        final_response.append({'type': 'default', 'texts': [origin_sentence]})
                        final_response.append({'type': 'quote', 'texts': [str(c) for c in sentence_citations]})
            else:
                if markdown:
                    final_response.append(origin_sentence)
                else:
                    final_response.append({'type': 'default', 'texts': [origin_sentence]})

            if line_break:
                if markdown:
                    final_response.append(line_break)
                else:
                    final_response.append({'type': 'default', 'texts': [line_break]})
        if markdown:
            final_response = ''.join(final_response)
        return final_response

    def concatenate_citations(self, result: list[dict] = None):
        """

        :param result:
        :return:
        """
        final_text = ""
        for item in result:
            if item['type'] == 'default':
                final_text += ''.join(item['texts'])
            elif item['type'] == 'quote':
                quotes = ''.join([f'[{q}]' for q in item['texts']])
                final_text += quotes
        return final_text


if __name__ == '__main__':
    mc = MatchCitation()

    result = mc.ground_response(
        response="巨齿鲨2是一部科幻冒险电影，由本·维特利执导，杰森·斯坦森、吴京、蔡书雅和克利夫·柯蒂斯主演。电影讲述了海洋霸主巨齿鲨，今夏再掀狂澜！乔纳斯·泰勒（杰森·斯坦森饰）与科学家张九溟（吴京饰）双雄联手，进入海底7000米深渊执行探索任务。他们意外遭遇史前巨兽海洋霸主巨齿鲨群的攻击，还将对战凶猛危险的远古怪兽群。惊心动魄的深渊冒险，巨燃巨爽的深海大战一触即发。",
        evidences=[
            "海洋霸主巨齿鲨，今夏再掀狂澜！乔纳斯·泰勒（杰森·斯坦森 饰）与科学家张九溟（吴京 饰）双雄联手，进入海底7000米深渊执行探索任务。他们意外遭遇史前巨兽海洋霸主巨齿鲨群的攻击，还将对战凶猛危险的远古怪兽群。惊心动魄的深渊冒险，巨燃巨爽的深海大战一触即发",
            "本·维特利 编剧：乔·霍贝尔埃里希·霍贝尔迪恩·乔格瑞斯 国家地区：中国 | 美国 发行公司：上海华人影业有限公司五洲电影发行有限公司中国电影股份有限公司北京电影发行分公司 出品公司：上海华人影业有限公司华纳兄弟影片公司北京登峰国际文化传播有限公司 更多片名：巨齿鲨2 剧情：海洋霸主巨齿鲨，今夏再掀狂澜！乔纳斯·泰勒（杰森·斯坦森 饰）与科学家张九溟（吴京 饰）双雄联手，进入海底7000米深渊执行探索任务。他们意外遭遇史前巨兽海洋霸主巨齿鲨群的攻击，还将对战凶猛危险的远古怪兽群。惊心动魄的深渊冒险，巨燃巨爽的深海大战一触即发……"
        ],
        selected_idx=[0, 1],
        markdown=True
    )

    print(result)

    # result = [
    #     {'type': 'default',
    #      'texts': ['巨齿鲨2是一部科幻冒险电影，由本·维特利执导，杰森·斯坦森、吴京、蔡书雅和克利夫·柯蒂斯主演。']},
    #     {'type': 'default', 'texts': ['电影讲述了海洋霸主巨齿鲨，今夏再掀狂澜']},
    #     {'type': 'quote', 'texts': ['2', '3']}, {'type': 'default', 'texts': ['！']},
    #     {'type': 'default',
    #      'texts': ['乔纳斯·泰勒（杰森·斯坦森饰）与科学家张九溟（吴京饰）双雄联手，进入海底7000米深渊执行探索任务']},
    #     {'type': 'quote', 'texts': ['2', '3']}, {'type': 'default', 'texts': ['。']},
    #     {'type': 'default', 'texts': ['他们意外遭遇史前巨兽海洋霸主巨齿鲨群的攻击，还将对战凶猛危险的远古怪兽群']},
    #     {'type': 'quote', 'texts': ['2', '3']}, {'type': 'default', 'texts': ['。']},
    #     {'type': 'default', 'texts': ['惊心动魄的深渊冒险，巨燃巨爽的深海大战一触即发']},
    #     {'type': 'quote', 'texts': ['2', '3']}, {'type': 'default', 'texts': ['。']}
    # ]
    # response=mc.concatenate_citations(result)
    # print(response)
