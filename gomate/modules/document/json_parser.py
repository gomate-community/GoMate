import json

import chardet

from gomate.modules.document.utils import find_codec


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
        sections = []
        try:
            sections.append(data['title'] +'\n'+data['content'])
        except:
            pass
        # print(len(sections),len(json_lines))
        return sections

    def crop(self, ck, need_position):
        raise NotImplementedError

    @staticmethod
    def remove_tag(txt):
        raise NotImplementedError


if __name__ == '__main__':
    jp = JsonParser()
    data = jp.parse(r'H:\Projects\GoMate\data\modified_demo.json')
    print(data[0])
