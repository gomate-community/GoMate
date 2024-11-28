#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:quincy qiang
# email:yanqiangmiffy@gmail.com
# datetime:2021/5/25 11:38
# description:"通用打印函数"
from libraries.timer import get_now_time


def usual_print(msg=None, prompt=None):
    if prompt:
        print("{}:{}------>：{}".format(get_now_time(), prompt, msg))
    else:
        print("{}:------>：{}".format(get_now_time(), msg))

def save_print(save_dir):
    print("{}:正在保存到路径------->：{}".format(get_now_time(), save_dir))

if __name__ == '__main__':
    usual_print('2021/05/25.csv',"保存文件")
    usual_print('2021/05/25.csv', )
