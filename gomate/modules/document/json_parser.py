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
            json_lines = txt.split('}\n')
            # print("json_parser", json_lines[0] + '}')
        else:
            with open(fnm, "r", encoding=get_encoding(fnm)) as f:
                txt = f.read()
        json_lines = json.loads(txt)
        # print(len(json_lines))
        # print("json_parser", json_lines[0] + '}')
        sections = []
        # for sec in txt.split("\n"):
        #     sections.append(sec)
        for line in json_lines:
            try:
                sections.append(line['title'] + line['content'])
            except:
                pass
        print(len(sections),len(json_lines))
        return sections

    def crop(self, ck, need_position):
        raise NotImplementedError

    @staticmethod
    def remove_tag(txt):
        raise NotImplementedError


if __name__ == '__main__':
    jp = JsonParser()
    data = jp.parse('/data/users/searchgpt/yq/GoMate_dev/data/docs/JSON格式/习语录1_list.json')
    print(data[0])
