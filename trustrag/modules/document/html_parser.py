# -*- coding: utf-8 -*-
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import chardet
import html_text
import readability

from trustrag.modules.document.utils import find_codec


def get_encoding(file):
    with open(file, 'rb') as f:
        tmp = chardet.detect(f.read())
        return tmp['encoding']


class HtmlParser:
    def parse(self, fnm):
        txt = ""
        if not isinstance(
            fnm, str):
            encoding = find_codec(fnm)
            txt = fnm.decode(encoding, errors="ignore")
        else:
            with open(fnm, "r", encoding=get_encoding(fnm)) as f:
                txt = f.read()

        html_doc = readability.Document(txt)
        title = html_doc.title()
        content = html_text.extract_text(html_doc.summary(html_partial=True))
        txt = f'{title}\n{content}'
        sections = txt.split("\n")
        return sections


if __name__ == '__main__':
    hp = HtmlParser()
    contents = hp.parse('/data/users/searchgpt/yq/GoMate_dev/data/docs/如何打通发展新质生产力的堵点卡点_新浪新闻.html')
    print(contents)
