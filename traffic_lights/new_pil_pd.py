# import cv2
#
# videoCapture = cv2.VideoCapture("/home/haoyu/yuhao_video/a827.avi")
#
#
# # fps = videoCapture.get()
# # size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
# #         int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
# #
#
# # videoWriter = cv2.VideoWriter('./data/video_plane.avi',)
#
# print(111)
#
# success, frame = videoCapture.read()
# num=0
# while 1:
#     # cv2.imshow("Oto Video", frame)  #
#     # cv2.waitKey(1000 / int(fps))  #
#     # videoWriter.write(frame)  #
#     #
#     cv2.imshow("imgs", frame)  #
#     cv2.waitKey(1)  #
#     # videoWriter.write(frame)  #
#     # if num%2==0:
#     #  cv2.imwrite('./imgs/{0}.jpg'.format(num), frame)
#     num+=1
#     success, frame = videoCapture.read()  #
#
import tensorflow as tf
import matplotlib.pyplot as plt
import time

import PIL.Image as Image
import numpy as np
import os
label_to_colours =    {0: [0, 0,0],
                       1: [128,0,0],
                       2: [ 0 ,28 ,0 ],
                       3: [128 ,128 ,0 ]
                     }

#
def class_to_img(input):
    new_tensor = input[:, :, :, [0]]
    # new_tensor=np.expand_dims(new_tensor,axis=-1)
    image_rgb = np.repeat(new_tensor, 3, axis=-1)
    for num in range(len(input)):
        shape=np.shape(input[num])
        for i in range(shape[0]):
            for j in range(shape[1]):
                cls_max=np.argmax(input[num][i][j] ,axis=0)
                image_rgb[num][i][j]=label_to_colours[cls_max]
                # print(cls_max)
    return image_rgb


# detector = Detector()0006
# path = "/home/haoyu/data_tracking_image_2/testing/image_02/0014"

path = "/home/llye/Desktop/imgcrop-ok/"#数据集合的目录
# path="../imgs22"
all_abs = []
for img_name in os.listdir(path):
    abs_img = os.path.join(path, img_name)
    all_abs.append(abs_img)

sort_abs_imgs = np.sort(all_abs)
print(sort_abs_imgs)
num = 0

globals_imgs_np=[]

for one_img in sort_abs_imgs:
    with Image.open(one_img) as im:
        num += 1
        print(num)
        #################尺寸变换
        image_resize = im.resize((128, 128))
        im_np = np.array(image_resize)
        globals_imgs_np.append(im_np)

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

plt.ion()

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    # output_graph_path = "../pb/road_old.pb"
    output_graph_path = "./lights_4cls.pb"
    # output_graph_path = "./road_t_bn_5w.pb"
    # output_graph_path = 'netsmodel/combined_modelok_pnet.pb'
    # 这里是你保存的文件的位置
    with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name="")
    # with tf.Session() as sess:


    with tf.Session().as_default() as sess:
        # print(a.eval())
    # print(b.eval(session=sess))

            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            input_x = sess.graph.get_tensor_by_name("Placeholder:0")
            # output = sess.graph.get_tensor_by_name("generator/BatchNorm_16/FusedBatchNorm:0")
            output = sess.graph.get_tensor_by_name("generator/add_10:0")
            # 这个是你保存文件的名字，取0是tensor
            # 输出的时候的名字
            # for im_np in globals_imgs_np:
            #         print(im_np)
            #         # plt.clf()
            #         a = time.time()
            #         # pre_img = sess.run(output, {input_x: [np.array(image) / 255 - 0.5]})
            #         pre_img = sess.run(output, {input_x: [im_np/255-0.5]})
            #
            #
            #         ccc = np.argmax(pre_img[0], axis=2)
            #         aaa = time.time()
            #
            #         ddd=np.multiply(im_np[:,:,2], ccc)
            #         # image = im_np
            #         ax2.imshow(ddd.astype(np.uint8))
            #         ax1.imshow(im_np.astype(np.uint8))
            #         plt.pause(0.02)
    # img1=ax1.imshow(im_np.astype(np.uint8))
    # img2=ax2.imshow(im_np.astype(np.uint8))
    num=0
    for im_np in globals_imgs_np[0:]:
            # print(im_np)
            # plt.clf()
            a = time.time()
            # pre_img = sess.run(output, {input_x: [np.array(image) / 255 - 0.5]})
            aa=time.time()
            pre_img = sess.run(output, {input_x: [im_np/255-0.5]})
            print(time.time()-aa)
            # output.eval(session=sess,input_x: [im_np/255-0.5])


            ccc = np.argmax(pre_img, axis=1)
            aaa = time.time()
            print(pre_img)
            num+=1
            if ccc==0:
                print("...............红色......................................")
                r_img=Image.fromarray(np.uint8(im_np))
                r_img.save("/home/llye/Desktop/red/{0}.jpg".format(num))
            if ccc==1:
                print(".................................绿色.........................................")
                r_img = Image.fromarray(np.uint8(im_np))
                r_img.save("/home/llye/Desktop/green/{0}.jpg".format(num))
            if ccc==2:
                print("....................................................黄色......................................")
                r_img = Image.fromarray(np.uint8(im_np))
                r_img.save("/home/llye/Desktop/yellow/{0}.jpg".format(num))
            if ccc == 3:
                print("..........................................................................................")
                r_img = Image.fromarray(np.uint8(im_np))
                r_img.save("/home/llye/Desktop/other/{0}.jpg".format(num))

            # ddd=np.multiply(im_np[:,:,2], ccc)
            # image = im_np
            # ax2.imshow(ddd.astype(np.uint8))
            # ax1.imshow(im_np.astype(np.uint8))
            # img1.set_data(im_np.astype(np.uint8))
            # img2.set_data(ddd.astype(np.uint8))
            # plt.pause(2)

    plt.clf()


# import  cv2
#
# cap = cv2.VideoCapture("./2222.mp4")
# print(cap)
#
#
# success, photo = cap.read()
# print(photo)
# while True:
#         # cv2.waitKey(1)  #
#         #
#         photo = cv2.resize(photo, (256, 540), fx=0.5, fy=0.5)
#         # print(np.shape(photo))
#         # aaa=pnet_detect(photo)
#         # b, g, r = cv2.split(photo)
#         # img = cv2.merge([r, g, b])
#         # im = Image.fromarray(img, "RGB")
#
#         # boxes = detector.detect(im)
#         # for box in boxes:
#         #     x1 = int(box[0])
#         #     y1 = int(box[1])
#         #     x2 = int(box[2])
#         #     y2 = int(box[3])
#         #     w = x2 - x1
#         #     h = y2 - y1
#         #     cv2.rectangle(photo, (x1, y1), (x2, y2), (0, 0, 255), 1)
#
#         cv2.imshow("capture", photo)
#         success, photo = cap.read()
#         if cv2.waitKey(100) & 0xFF == ord('q'):
#             break