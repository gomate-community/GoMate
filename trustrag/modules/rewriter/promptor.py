WEB_SEARCH_EN = """Please write a passage to answer the question.
Question: {}
Passage:"""

WEB_SEARCH_ZH = """请写一段文字来回答这个问题。
问题: {}
答案："""

SCIFACT_EN = """Please write a scientific paper passage to support/refute the claim.
Claim: {}
Passage:"""

SCIFACT_ZH = """请写一段科学论文来支持或反驳这个论点。
论点: {}
答案："""

ARGUANA_EN = """Please write a counter argument for the passage.
Passage: {}
Counter Argument:"""

ARGUANA_ZH = """请针对这段论述写出一个反驳论点。
论述: {}
反驳论点:"""

TREC_COVID_EN = """Please write a scientific paper passage to answer the question.
Question: {}
Passage:"""

TREC_COVID_ZH = """请写一段科学论文来回答这个问题。
问题: {}
答案："""

FIQA_EN = """Please write a financial article passage to answer the question.
Question: {}
Passage:"""

FIQA_ZH = """请写一段金融文章来回答这个问题。
问题: {}
答案："""

DBPEDIA_ENTITY_EN = """Please write a passage to answer the question.
Question: {}
Passage:"""

DBPEDIA_ENTITY_ZH = """请写一段文字来回答这个问题。
问题: {}
答案："""

TREC_NEWS_EN = """Please write a news passage about the topic.
Topic: {}
Passage:"""

TREC_NEWS_ZH = """请写一段新闻来介绍这个话题。
话题: {}
答案："""

MR_TYDI_EN = """Please write a passage in {} to answer the question in detail.
Question: {}
Passage:"""

MR_TYDI_ZH = """请用{}语言写一段详细回答这个问题的文字。
问题: {}
答案："""


class Promptor:
    def __init__(self, task: str, language: str = 'en'):
        self.task = task
        self.language = language

    def build_prompt(self, query: str):
        lang_suffix = '_EN' if self.language == 'en' else '_ZH'
        try:
            prompt_template = globals()[self.task.upper().replace('-', '_') + lang_suffix]
            return prompt_template.format(self.language, query) if self.task == 'mr-tydi' else prompt_template.format(
                query)
        except KeyError:
            raise ValueError('Task or language not supported')


if __name__ == '__main__':
    # Usage example:
    promptor = Promptor('WEB_SEARCH', 'zh')
    print(promptor.build_prompt("北京流行什么？"))
