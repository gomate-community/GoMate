import os
import shutil
import win32com.client as win32
import  time
# 获取当前文件夹路径
current_dir = os.getcwd()
target_dir = os.path.join(current_dir, 'documents')

# 确保目标文件夹存在
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Word转换的函数
def convert_doc_to_docx(doc_path):
    word = win32.Dispatch('Word.Application')
    doc = word.Documents.Open(doc_path)
    docx_path = doc_path.replace(".doc", ".docx")
    doc.SaveAs(docx_path, FileFormat=16)  # 16是Word的docx格式代码
    time.sleep(10)
    doc.Close()
    word.Quit()
    return docx_path

# 遍历当前文件夹所有文件
for root, dirs, files in os.walk(current_dir):
    for file in files:
        file_path = os.path.join(root, file)
        destination_path = os.path.join(target_dir, file)

        if file.lower().endswith('.doc'):
            # 检查是否已经有对应的.docx文件
            docx_file = file.replace(".doc", ".docx")
            if docx_file not in files:
                # 如果不存在对应的.docx文件，则进行转换
                print(f"Converting {file} to .docx")
                converted_docx_path = convert_doc_to_docx(file_path)
                # 移动转换后的文件到documents文件夹
                if not os.path.exists(os.path.join(target_dir, os.path.basename(converted_docx_path))):
                    shutil.copy(converted_docx_path, target_dir)
            else:
                print(f"{file} already has a corresponding .docx file, skipping conversion.")

        # 如果文件是docx或者pdf，直接移动
        elif file.lower().endswith('.docx') or file.lower().endswith('.pdf'):
            # 如果文件已经在目标目录，不再移动
            if not os.path.exists(destination_path):
                print(f"Moving {file} to documents folder.")
                shutil.copy(file_path, target_dir)
            else:
                print(f"{file} already exists in the documents folder, skipping copy.")

print("Process completed.")