from tqdm import tqdm
import chardet

def get_encoding(file):
    with open(file, 'rb') as f:
        tmp = chardet.detect(f.read())
        return tmp['encoding']

class TextParser(object):
    def parse(self, fnm, encoding="utf-8", from_page=0, to_page=100000, **kwargs):
        # 如果 fnm 不是字符串（假设是字节流等），则使用 find_codec 找到编码
        if not isinstance(fnm, str):
            encoding = get_encoding(fnm) if encoding is None else encoding
            txt = fnm.decode(encoding, errors="ignore")
        else:
            # 如果是字符串文件路径，且没有传入 encoding，则自动调用 get_encoding 检测编码
            encoding = get_encoding(fnm) if encoding is None else encoding
            with open(fnm, "r", encoding=encoding) as f:
                txt = f.read()

        sections = []
        # 按行进行解析并显示进度
        for sec in tqdm(txt.split("\n"), desc="Parsing"):
            if sec.strip():
                sections.append(sec)
        return sections

    def crop(self, ck, need_position):
        raise NotImplementedError

    @staticmethod
    def remove_tag(txt):
        raise NotImplementedError