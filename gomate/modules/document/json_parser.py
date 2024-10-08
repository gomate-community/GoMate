import json

import chardet

from gomate.modules.document.utils import find_codec
from gomate.modules.document.utils import PROJECT_BASE

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
            sections.append(data['title'] +'\n'+data['content'])
        except:
            if 'documents' in data:
                for document in data['documents']:
                    for section in document['sections']:
                        sections.append(section['sec_theme'] + '\n' + section['content'])
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
