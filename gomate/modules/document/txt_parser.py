import chardet
from tqdm import tqdm
from gomate.modules.document.utils import find_codec


def get_encoding(file):
    with open(file, 'rb') as f:
        tmp = chardet.detect(f.read())
        return tmp['encoding']


class TextParser(object):
    def parse(self, fnm, from_page=0, to_page=100000, **kwargs):
        if not isinstance(fnm, str):
            encoding = find_codec(fnm)
            txt = fnm.decode(encoding, errors="ignore")
        else:
            with open(fnm, "r", encoding=get_encoding(fnm)) as f:
                txt = f.read()

        sections = []
        for sec in tqdm(txt.split("\n"),desc="Parsing"):
            sections.append(sec)
        return sections

    def crop(self, ck, need_position):
        raise NotImplementedError

    @staticmethod
    def remove_tag(txt):
        raise NotImplementedError
