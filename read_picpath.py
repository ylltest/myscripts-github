#!/usr/bin/env python
# coding:utf-8
import os

pic = {}
path = input('Dir_name:')

def pic_path(path):
    domain = os.path.abspath(path)
    for i in os.listdir(path):
        file_path = os.path.join(domain, i)  # 文件路径
        if os.path.isdir(file_path):
            pic_path(file_path)
        else:
            file_name = file_path[len(path) + 1:-4]  # 文件名
            key, value = file_name, file_path
            pic[key] = value

pic_path(path)