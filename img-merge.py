#!/usr/bin/env python
# coding:utf-8
from PIL import Image
#加载底图
base_img = Image.new(mode='RGBA', size=(1241, 752))
# 可以查看图片的size和mode，常见mode有RGB和RGBA，RGBA比RGB多了Alpha透明度
# print base_img.size, base_img.mode
box1 = (0, 0, 1241, 376)  # 底图上需要P掉的区域
box2 = (0, 376, 1241, 752)
#加载需要P上去的图片
def merge_pic():
    num = 0
    for i in range(0, 574):
        num += 1
        tmp_img1 = Image.open('/home/llye/Desktop/test/0020-ok-jpg-re/{0}.jpg'.format(num))
        tmp_img2 = Image.open('/home/llye/Desktop/test/tepimg-ok-jpg-re/{0}.jpg'.format(num))
        #这里可以选择一块区域或者整张图片
        # region = tmp_img.crop((0,0,100,100)) #选择一块区域
        #或者使用整张图片
        region1 = tmp_img1
        region2 = tmp_img2

        #使用 paste(region, box) 方法将图片粘贴到另一种图片上去.
        # 注意，region的大小必须和box的大小完全匹配。但是两张图片的mode可以不同，合并的时候回自动转化。如果需要保留透明度，则使用RGMA mode
        #提前将图片进行缩放，以适应box区域大小
        # region = region.rotate(180) #对图片进行旋转
        region1 = region1.resize((box1[2] - box1[0], box1[3] - box1[1]))
        base_img.paste(region1, box1)
        region2 = region2.resize((box2[2] - box2[0], box2[3] - box2[1]))
        base_img.paste(region2, box2)
        #base_img.show() # 查看合成的图片
        base_img.save('/home/llye/Desktop/test/out/{0}.jpg'.format(num)) #保存图片

merge_pic()