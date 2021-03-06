#!/usr/bin/env python
# coding:utf-8
import os,shutil
from PIL import Image
path = input('Path:')   # 待筛选图片路径(不加"\")
dir_name = input('Dir_name:')   # 待筛选图片目录名
size_argument = float(input('Proportion:'))  # 精度
abnormal_img=[]
def find_abnormal_img():
    for file in os.listdir(path):
        domain = os.path.abspath(path)
        file = os.path.join(domain, file)
        print(file) #打印文件路径
        img = Image.open(file)
        print(img.size)
        file_name = file[len(path)+1:-4]
        img_size = img.size
        x=img_size[0]
        y=img_size[1]
        res=float(y/x)
        # print("%.2f"%res)
        if res>size_argument:
            print('异常图片:'+file_name)
            abnormal_img.append(file_name)
        else:print('正常图片')
        img.close()

def copy_abnormal_img():
    domain = os.path.abspath(path)
    new_domain = domain[:-len(dir_name)]
    new_dir = os.mkdir(os.path.join(new_domain, 'abnormal_img'))
    for file in os.listdir(path):
        source_path = os.path.join(domain, file)
        source_name = source_path[len(path):-4]
        target_path = new_domain+'\\abnormal_img\\'+source_name+'.jpg'
        img = Image.open(source_path)
        img_size = img.size
        x=img_size[0]
        y=img_size[1]
        res=float(y/x)
        if res>size_argument:
            shutil.copy(source_path, target_path)
            print('图片保存成功')
        else:print('正常图片')
        img.close()

def save_img_number():
    for img in abnormal_img:
        new_path = path[:-len(dir_name)]
        f = open(new_path+r'\abnormal_nmuber.txt', 'a')
        f.write(img + '\n')
        print('保存异常图片编号:'+ img)
        f.close()

find_abnormal_img()
copy_abnormal_img()
save_img_number()