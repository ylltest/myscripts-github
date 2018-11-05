#!/usr/bin/env python
# coding:utf-8
from ctypes import *
import random
import cv2,time,os
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString

img_number = 1

class Info:
    pass

def createxml(filename, imagePath, imageName, width, height, info):
    xmlfile = open(filename, "w")
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = imagePath
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = imageName
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    for i in info:
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = i.name
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(i.xmin)
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(i.ymin)
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(i.xmax)
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(i.ymax)

    xml = tostring(node_root, pretty_print=True)  #格式化显示，该换行的换行
    dom = parseString(xml)
    dom.writexml(xmlfile, encoding="utf-8")
    xmlfile.close()

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def savepicture(img, path, name):
    cv2.imwrite(path+name, img)

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    global img_number
    origimg = cv2.imread(image)
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    wh = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                xmin = int(b.x - b.w/2)
                xmax = int(b.x + b.w/2)
                ymin = int(b.y - b.h/2)
                ymax = int(b.y + b.h/2)
                p1 = (xmin, ymin)
                p2 = (xmax, ymax)
                coords = (xmin, ymin),(xmax, ymax)
                if (meta.names[i] == "traffic light"):
                    crop_img = origimg[p1[1]:p2[1], p1[0]:p2[0]]
                    savepicture(crop_img, crop_pic_dir, str(img_number) + ".jpg")
                    img_number = img_number + 1
                    print coords
                    info = Info()
                    info.name = "traffic light"
                    info.xmin = xmin
                    info.ymin = ymin
                    info.xmax = xmax
                    info.ymax = ymax
                    res.append(info)
                    w = int(b.w)
                    h = int(b.h)
                    wh.append(w)
                    wh.append(h)
                    # cv2.rectangle(origimg, p1, p2, (0, 255, 0))
                    # res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))  #类名,可信度,(中心点x值，中心点y值,宽度,高度)
    xml_name = crop_xml_dir + os.path.splitext(os.path.basename(image))[0] + ".xml"
    print "xml_name=" + xml_name
    if len(wh) != 0:
        width = wh[0]
        hight = wh[1]
        createxml(xml_name, crop_pic_dir, image, width, hight, res)
    # cv2.imshow("YOLOV3", origimg)
    # cv2.waitKey(0)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     free_image(im)
    #     free_detections(dets, num)
    free_image(im)
    free_detections(dets, num)
    
if __name__ == "__main__":
    # net = load_net("cfg/yolov3-tiny.cfg", "yolov3-tiny.weights", 0)
    net = load_net("cfg/yolov3.cfg", "yolov3.weights", 0)
    meta = load_meta("cfg/coco.data")
    time.sleep(1)
    input_image_dir = raw_input('input_image_dir:')
    crop_xml_dir = raw_input('crop_xml_dir:')
    crop_pic_dir = raw_input('crop_pic_dir:')
    # input_image_dir = "/home/llye/Desktop/a/"
    # crop_xml_dir = "/home/llye/Desktop/xml/"
    # crop_pic_dir = "/home/llye/Desktop/b/"
    for f in os.listdir(input_image_dir):
        if detect(net, meta, input_image_dir + f) == False:
            break