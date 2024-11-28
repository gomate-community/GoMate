import json
import os
import re

from trustrag.modules.document.utils import PROJECT_BASE


def process_markdown(markdown_content):
    # Split the content into sections based on h1 headers
    sections = re.split(r'\n# ', markdown_content)

    contents = []

    for section in sections[1:]:  # Skip the first empty section
        # Split the section into title and content
        parts = section.split('\n', 1)

        if len(parts) == 1:
            # Case: # followed by no content
            continue

        title = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else ''

        # Remove image tags ![](images...) from the text
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

        if title and (text or text == ''):
            contents.append({'headline': title, 'text': text})

    return contents


def merge_short_content(contents):
    merged_contents = []
    i = 0
    while i < len(contents):
        current = contents[i]

        # If the text is empty and there is a next content, merge it with the next one
        if current['text'] == '' and i + 1 < len(contents):
            next_content = contents[i + 1]
            merged_content = {
                'headline': current['headline'] + ' ' + next_content['headline'],
                'text': next_content['text']
            }
            merged_contents.append(merged_content)
            i += 2  # Skip the next content as it's merged
        else:
            merged_contents.append(current)
            i += 1

    return merged_contents


def convert2json(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            if file_path.endswith('_ocr.md') or file_path.endswith('.md'):
                filename = filename.replace('_ocr.md', '').replace('.md', '')
                with open(file_path, 'r', encoding='utf-8') as file:
                    markdown_content = file.read()
                    contents = process_markdown(markdown_content)
                    json_output = {'title': filename, 'metadata': filename, 'source': filename, 'date': '2024-09-12',
                                   'contents': contents, 'content': markdown_content}
                    with open(f'{PROJECT_BASE}/data/competitions/df/jsons/{filename}.json', 'w',
                              encoding='utf-8') as json_file:
                        contents = merge_short_content(json_output['contents'])
                        json_output['contents'] = contents
                        json.dump(json_output, json_file, ensure_ascii=False, indent=4)


import os


def convert2text(directory):
    texts = []
    # 遍历文件
    for root, dirs, files in os.walk(directory):
        # 先创建一个集合来存储文件名，不含后缀，用于判断是否有相应的 _ocr.md 文件
        seen_files = set()

        # 优先查找 _ocr.md 文件
        for filename in files:
            if filename.endswith('_ocr.md'):
                base_filename = filename.replace('_ocr.md', '')  # 去掉后缀
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    markdown_content = file.read()
                    texts.append(markdown_content)
                seen_files.add(base_filename)  # 标记已经读取该文件名对应的 _ocr.md 文件
                with open(f'{PROJECT_BASE}/data/competitions/df/texts/{base_filename}.txt', 'w', encoding='utf-8') as f:
                        for line in markdown_content.split('\n'):
                            if line.startswith('#'):
                                line=line[1:]
                            f.write(line+'\n')
        # 查找 .md 文件（但不处理已存在 _ocr.md 的文件）
        for filename in files:
            if filename.endswith('.md') and not filename.endswith('_ocr.md'):
                base_filename = filename.replace('.md', '')  # 去掉后缀
                if base_filename not in seen_files:  # 如果没有对应的 _ocr.md 文件
                    file_path = os.path.join(root, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        markdown_content = file.read()
                        texts.append(markdown_content)
                    with open(f'{PROJECT_BASE}/data/competitions/df/texts/{base_filename}.txt', 'w',
                              encoding='utf-8') as f:
                        for line in markdown_content.split('\n'):
                            if line.startswith('#'):
                                line = line[1:]
                            f.write(line + '\n')

directory = f'{PROJECT_BASE}/data/competitions/df/output'  # 替换为你的目录路径
convert2text(directory)
