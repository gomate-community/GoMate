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

    def ground_response(
            self,
            response: str,
            evidences: List[str],
            selected_idx: List[int],
            markdown: bool = True,
            show_code=True
    ):
        sentences = self.cut(response)
        final_response = []
        print(response)
        print(evidences)
        print(selected_idx)
        selected_idx=[i-1 for i in selected_idx]
        print(selected_idx)
        print(len(evidences))
        for sentence in sentences:
            if not sentence.strip():
                continue

            sentence_seg_cut = set(jieba.lcut(self.remove_stopwords(sentence)))
            sentence_seg_cut_length = len(sentence_seg_cut)

            best_match = None
            best_ratio = 0
            best_idx = None

            for i,idx in enumerate(selected_idx):
                evidence = evidences[i]
                evidence_sentences = self.cut(evidence)

                for evidence_sentence in evidence_sentences:
                    evidence_seg_cut = set(jieba.lcut(self.remove_stopwords(evidence_sentence)))
                    overlap = sentence_seg_cut.intersection(evidence_seg_cut)
                    ratio = len(overlap) / sentence_seg_cut_length

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match = evidence_sentence
                        best_idx = idx + 1

            threshold = 0.7 if len(sentence) > 20 else 0.6

            if best_ratio > threshold:
                final_response.append(f"{sentence}。[{best_idx}]")
                if show_code:
                    final_response.append(f"\n```\n{best_match}。\n```\n")
            else:
                final_response.append(f"{sentence}。")

        return ''.join(final_response)


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


