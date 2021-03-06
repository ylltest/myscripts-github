import xml.etree.ElementTree as ET
import cv2
import os

box = []

input_image_dir="C:\\Users\\Dell\Desktop\\test\\img"
crop_xml_dir="C:\\Users\\Dell\Desktop\\test\\xml\\"
crop_dir="C:\\Users\\Dell\Desktop\\test\\crop\\"

def savepicture(img, path, name):
    cv2.imwrite(path+name, img)

def crop_Image(imagefile):
    global box
    origimg = cv2.imread(imagefile)
    tree=ET.parse(crop_xml_dir + xmi_name + ".xml")
    root=tree.getroot()
    for node in root.iter('bndbox'):
        for i in node:
            box.append(i.text)
    xmin = int(box[0])
    ymin = int(box[1])
    xmax = int(box[2])
    ymax = int(box[3])
    p1 = (xmin, ymin)
    p2 = (xmax, ymax)
    crop_img = origimg[p1[1]:p2[1], p1[0]:p2[0]]
    savepicture(crop_img, crop_dir, str(img_name) + ".jpg")
    print(img_name)
    box = []

for f in os.listdir(input_image_dir):
    img_name = str(f[:-4])
    xmi_name = img_name
    if crop_Image(input_image_dir+"\\"+f) == False:
        break