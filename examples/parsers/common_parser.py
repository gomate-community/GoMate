from gomate.modules.document.common_parser import CommonParser


cp=CommonParser()
content=cp.parse('/data/users/searchgpt/yq/GoMate_dev/data/docs/Agent.docx')
print(content)