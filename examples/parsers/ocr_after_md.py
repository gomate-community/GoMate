from paddleocr import PaddleOCR
from pathlib import Path


# import logging

# # 配置日志记录器，设置日志级别为INFO
# logging.basicConfig(level=logging.INFO)

import re 

import click


### 下面是简单的版面整理 函数 ### 
threshold_value = 12
# 聚类成行
def cluster_into_lines(ocr_lines):
    lines = []
    current_line = [ocr_lines[0]]
    for i in range(1, len(ocr_lines)):
        current_block, current_line_height = current_line[-1]
        next_block, next_line_height = ocr_lines[i]
        # 确定是否在同一行
        
        if next_block[-1] - current_block[-1] < threshold_value:  # 50是行间距的阈值，可以根据实际情况调整
            current_line.append((next_block, next_line_height))
        else:
            lines.append(current_line)
            current_line = [(next_block, next_line_height)]
    lines.append(current_line)  # 添加最后一行
    return lines

# 对每个区块内的文本按列排序
def sort_by_column(blocks):
    #sorted(blocks,key=lambda x: x[0][1],reverse=True)
    for block in blocks:
         block.sort(key=lambda x: x[0][0])  # 按x坐标排序
    return blocks

# 主处理函数
def process_ocr_output(ocr_output):
    # 按y坐标对OCR结果进行排序
    ocr_output.sort(key=lambda x: x[0][1])
    # 聚类成行
    blocks = cluster_into_lines(ocr_output)
    # 对每个区块内的文本按列排序
    sorted_blocks = sort_by_column(blocks)
    return sorted_blocks

### 上面是简单的版面整理 函数 ### 



# 加载OCR模型
ocr = PaddleOCR(use_angle_cls=True, 
            lang="japan",
            page_num  = 100 ,
            use_gpu=True,
            show_log=False,
            det_model_dir = "PDFOCR_Paddle/japan_PP-OCRv3/detection",
            rec_model_dir="PDFOCR_Paddle/japan_PP-OCRv3/recognize",
            )  

## 从单个图片里识别文字的函数。 注意到 fpath 是.jpg文件。
def single_ocr_recognize_frompath(fpath):
    fpath = Path(fpath)
    result = ocr.ocr(str(fpath), cls=True)

    
    the_page_string = ""
    if result[0] is not None:
        thepageboxes = [ (x[0][0] , x[1][0] ) for x in result[0]]
        thepageboxes = process_ocr_output(ocr_output=thepageboxes)
        
        for j, block in enumerate(thepageboxes, 1):
            #print(f"区块 {j}:")             
            for text_block in block:
                #print(f"  文本: {text_block[1]}，坐标: {text_block[0]}")
                the_page_string += text_block[1] + " "
            the_page_string += "\n"
    return the_page_string

# 从md格式里，拿出图片路径
"""
输入下面这样形式的字符串：![](images/089c0470d201c19f2a17503966789950369e263e2d142fca176612ce4f540ddf.jpg)  ，
输出其中的 括号内的字符串：images/089c0470d201c19f2a17503966789950369e263e2d142fca176612ce4f540ddf.jpg
"""
def extract_content(s):
    # 定义正则表达式模式
    pattern = r'!\[\]\((.*?)\)'
    
    # 使用正则表达式进行匹配
    match = re.search(pattern, s)
    
    # 如果找到匹配项，返回括号内的内容；否则返回 None
    return match.group(1) if match else None



## 读取 magic pdf 的输出文件夹，识别里面的md文件的图片
@click.command()
@click.option('--ocrdir',
            default="",
            help='The output path of magic pdf ')
def ocr_all_mdfiles(ocrdir):

    if ocrdir == "":
        print("ocr目标目录为空！程序结束")
        exit()
    else:
        print("-"*10 + "开始对magic pdf 的结果 进一步ocr" + "-"*10)
        ocrdir = Path(ocrdir)
        subfiles = [ x  for x in ocrdir.glob("*") if x.is_dir()]
        tnum = len(subfiles)
        k1 = 0
        k2 = 0
        for subfile in subfiles:    
            k1 +=1 
            subfilestem = subfile.stem

            # (1)找到md文件
            md_path = subfile / "auto" / f"{subfilestem}.md"


            # (2) 招到img文件
            if (subfile / "auto" / "images").exists():
                images_paths = [ x for x in (subfile / "auto" / "images").rglob("*.jpg")]
            else:
                images_paths = []
            # (3) 若有图片就调用 ocr 识别。
            if len(images_paths) >= 1 :
                timgs = len(images_paths)
                ## 3.1 逐行读取 md 文件 找出 ![ 开头的行。 
                with open(md_path, 'r', encoding='utf-8') as md_file:
                    mdcontent_lines = [ x for x in md_file.readlines()] 
                    img_marks = [   (extract_content(value),index)   for index, value in enumerate(mdcontent_lines) if value.startswith("![") ]
                    ## ![](images/089c0470d201c19f2a17503966789950369e263e2d142fca176612ce4f540ddf.jpg)  
                    ## 3.2 调用 ocr 识别图片。
                    k2 = 0
                    for item in img_marks:
                        k2+=1
                        img_path,index = item
                        img_path = subfile / "auto" / img_path
                        ocr_out = single_ocr_recognize_frompath(fpath=img_path)

                        mdcontent_lines[index] = ocr_out
                        print(f"识别第({k1}|{tnum})个的第({k2}|{timgs})个jpg.")

                    ## 识别结束后，写入新的md文件到txt

                    newtxtpath = subfile / "auto" / (subfilestem + ".txt")

                    with open(newtxtpath,'w',encoding="utf-8") as f3:
                        outstring = "".join(mdcontent_lines)

                        # 3.2 清除其他md格式

                        outstring = re.sub(r'\$.*?\$', '', outstring)
                        f3.write(outstring)
                print(f"finish ocr")
            else:
                pass
    print("finish all ocr")


    pass







if __name__ =="__main__":

    ocr_all_mdfiles()

    pass