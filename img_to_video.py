#!/usr/bin/env python
# coding:utf-8
import cv2
# img_root = '/home/llye/Desktop/test/out-merge/'#这里写你的文件夹路径，比如：/home/youname/data/img/,注意最后一个文件夹要有斜杠
img_root = input('img_path:')
fps = 15 #保存视频的FPS，可以适当调整
size=(1241,752)#保存图片的尺寸必须与原图片一致，否则转换后视频无法播放
#可以用(*'DVIX')或(*'X264'),如果都不行先装ffmpeg: sudo apt-get install ffmpeg
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter('/home/llye/Desktop/test/video/ok.avi',fourcc,fps,size)
for i in range(0,574):  #这里的循环值与要转换的图片数量保持一致，我这里是400张
    frame = cv2.imread(img_root+str(i)+'.jpg')
    videoWriter.write(frame)
videoWriter.release()
