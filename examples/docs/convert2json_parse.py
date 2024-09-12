import json
import os
import re


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


def list_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            if file_path.endswith('_ocr.md'):
                filename = filename.replace('_ocr.md', '')
                with open(file_path, 'r', encoding='utf-8') as file:
                    markdown_content = file.read()
                    contents = process_markdown(markdown_content)
                    json_output = {'title': filename, 'metadata': filename, 'source': filename, 'date': '2024-09-12',
                                   'contents': contents, 'content': markdown_content}
                    with open(f'jsons/{filename}.json', 'w', encoding='utf-8') as json_file:
                        json.dump(json_output, json_file, ensure_ascii=False)


directory = 'processed'  # 替换为你的目录路径
list_all_files(directory)
