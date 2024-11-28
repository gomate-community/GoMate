import json

import chardet

from trustrag.modules.document.utils import PROJECT_BASE
from trustrag.modules.document.utils import find_codec


def get_encoding(file):
    with open(file, 'rb') as f:
        tmp = chardet.detect(f.read())
        return tmp['encoding']


class JsonParser(object):
    def parse(self, fnm, from_page=0, to_page=100000, **kwargs):
        if not isinstance(fnm, str):
            encoding = find_codec(fnm)
            txt = fnm.decode(encoding, errors="ignore")
        else:
            with open(fnm, "r", encoding=get_encoding(fnm)) as f:
                txt = f.read()
        # print(txt)
        data = json.loads(txt)
        # print(data)
        sections = []
        try:
            sections.append(
                {
                    'source': data['source'],
                    'title': data['title'],
                    'date': data['date'],
                    'sec_num': 0,
                    'content': data['title'] + '\n' + data['content'],
                }
            )
        except:
            if 'sections' in data:
                # for document in data['documents']:
                for section in data['sections']:
                    sections.append(
                        {
                            'source': data['file_name'],
                            'title': data['title'],
                            'date': data['date'],
                            'sec_num': section['sec_num'],
                            'content': section['sec_theme'] + '\n' + section['content'],
                            'chunks': [
                                section['sec_theme'] + '\n' +
                                chunk['content'] for chunk in section['chunks']
                            ]
                        }
                    )
        # print(len(sections),len(json_lines))
        return sections

    def crop(self, ck, need_position):
        raise NotImplementedError

    @staticmethod
    def remove_tag(txt):
        raise NotImplementedError


if __name__ == '__main__':
    jp = JsonParser()
    print(f'{PROJECT_BASE}/data/docs/final_data/《中办通报》.json')
    data = jp.parse(f'{PROJECT_BASE}/data/docs/final_data/《中办通报》.json')
    print(data[0])
