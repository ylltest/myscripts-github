#!/usr/bin/env python
# coding:utf-8
import cv2, os
import glob
import sys, getopt

inputPath = input('inputPath:')
outputPath = input('outputPath:')
skipNum = 9
count = 1

def main(argv):
    global inputPath,outputPath,skipNum,count

    opts, args = getopt.getopt(argv,"hi:o:s:c",["ipath=","opath=","skip=", "count"])

    for op, value in opts:
        if op == "-i":
            inputPath = value
        elif op == "-o":
            outputPath = value
        elif op == "-s":
            skipNum = value
        elif op == "-c":
            count = value
        elif op == "-h":
            usage()
            sys.exit()

    count = 1

    print(inputPath + "/*/*")
    for pathAndFilename in glob.iglob(os.path.join(inputPath, "*.mp4")):
        print(pathAndFilename)
        processVideo(pathAndFilename, outputPath)

def usage():
   print('video2pic.py -i <inputPath> -o <outputPath> -s <skipNum>')


def processVideo(video, path):
    global count
    print("video=" + video + " path=" + path)
    print(count)
    c = 1
    vc=cv2.VideoCapture(video)
    if vc.isOpened():
        rval,frame=vc.read()
    else:
        rval=False
    while rval:
        rval, frame = vc.read()
        if c % (skipNum) == 0:
            cv2.imwrite(path + str(count) + '.jpg', frame)
            count = count + 1
        c=c+1
    vc.release()

if __name__ == "__main__":
    main(sys.argv[1:])