import re
import json


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
            contents.append({'title': title, 'text': text})

    return contents


def main():
    # Read the markdown file
    with open('temp/output/00-计算所工作人员年休假实施办法-发文/00-计算所工作人员年休假实施办法-发文_ocr.md', 'r', encoding='utf-8') as file:
        markdown_content = file.read()

    # Process the markdown content
    contents = process_markdown(markdown_content)

    # Create the JSON structure
    json_output = {'contents': contents}

    # Write the JSON to a file
    with open('output.json', 'w', encoding='utf-8') as file:
        json.dump(json_output, file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
