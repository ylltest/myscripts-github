#!/usr/bin/env python
# coding:utf-8
from PIL import Image
#加载底图
base_img = Image.open('/home/llye/Desktop/test/new/1.png')
# 可以查看图片的size和mode，常见mode有RGB和RGBA，RGBA比RGB多了Alpha透明度
# print base_img.size, base_img.mode
box = (166, 64, 320, 337)  # 底图上需要P掉的区域

#加载需要P上去的图片
tmp_img = Image.open('/home/llye/Desktop/test/new/2.png')
#这里可以选择一块区域或者整张图片
region = tmp_img.crop((0,0,304,546)) #选择一块区域
#或者使用整张图片
# region = tmp_img

#使用 paste(region, box) 方法将图片粘贴到另一种图片上去.
# 注意，region的大小必须和box的大小完全匹配。但是两张图片的mode可以不同，合并的时候回自动转化。如果需要保留透明度，则使用RGMA mode
#提前将图片进行缩放，以适应box区域大小
# region = region.rotate(180) #对图片进行旋转
region = region.resize((box[2] - box[0], box[3] - box[1]))
base_img.paste(region, box)
#base_img.show() # 查看合成的图片
base_img.save('/home/llye/Desktop/test/new/out.png') #保存图片