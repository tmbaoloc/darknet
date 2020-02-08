from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import os
import glob


# Ham sap xep contour tu trai sang phai tu tren xuong duoi

def sort_contours(cnts):
    list1=[]
    list2=[]
    for c in cnts:
        x, y, w, h=cv2.boundingRect(c)        
        if y<75:
            list1.append(c)            
        else:
            list2.append(c)            
    sorted_list1 = sorted(list1, key=lambda ctr: cv2.boundingRect(ctr)[0])
    sorted_list2 = sorted(list2, key=lambda ctr: cv2.boundingRect(ctr)[0])
    cnts= sorted_list1 +sorted_list2 
    return cnts

# Dinh nghia cac ky tu tren bien so
char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'
def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax
# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
        else:
            newString += '9'
    return newString
def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        image1 = img[ymin:ymax,xmin:xmax]
        
        if image1.size !=0:
            
            digit_w = 30 # Kich thuoc ki tu
            digit_h = 60 # Kich thuoc ki tu
            image1 = cv2.resize(image1, (300,200), 1)
            roi = image1
            model_svm = cv2.ml.SVM_load('svm.xml')
            #continue
            gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('Gray', gray)
            #binary=cv2.threshold(gray, 127, 255,
                        #cv2.THRESH_BINARY_INV)[1]
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 85, 10)
            cv2.imshow('Binary', binary)
            #kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            #thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
            _, cont, _= cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            plate_info = ""

            
            for c in sort_contours(cont):
                (x, y, w, h) = cv2.boundingRect(c)    
                ratio = h/w
                if 1.5<=ratio<=3.5: # Chon cac contour dam bao ve ratio w/h
                    if 0.3<=h/roi.shape[0]<=0.8: 
                        
                        # Ve khung chu nhat quanh so
                        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                                
                        # Tach so va predict
                        curr_num = binary[y:y+h,x:x+w]
                        curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                        _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                        
                        curr_num = np.array(curr_num,dtype=np.float32)
                        curr_num = curr_num.reshape(-1, digit_w * digit_h)
                        
                        # Dua vao model SVM
                        result = model_svm.predict(curr_num)[1]
                        result = int(result[0, 0])

                        if result<9: # Neu la so thi hien thi luon
                            result = str(result)
                        else: #Neu la chu thi chuyen bang ASCII
                            result = chr(result)

                        plate_info +=result
            cv2.imshow("Cac contour tim duoc", roi)
            print("Bien so=", fine_tune(plate_info))
            
            cv2.putText(img,
                        " [" + fine_tune(plate_info) + "]",
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        [25, 25, 225], 2)
        else: 
            continue
    return img
# Đường dẫn ảnh
#img_path = "./test/IMG.jpg"
path = "./test/*.jpg"
#dir = os.listdir(path)
#img_path = [cv2.imread(file) for file in glob.glob("path/to/files/*.png")]
netMain = None
metaMain = None
altNames = None




configPath = "./LP/yolov3-tiny_obj.cfg"
weightPath = "./yolov3-tiny_obj_4000.weights"
metaPath = "./LP/LP.data"
if not os.path.exists(configPath):
    raise ValueError("Invalid config path `" +
                        os.path.abspath(configPath)+"`")
if not os.path.exists(weightPath):
    raise ValueError("Invalid weight path `" +
                        os.path.abspath(weightPath)+"`")
if not os.path.exists(metaPath):
    raise ValueError("Invalid data file path `" +
                        os.path.abspath(metaPath)+"`")
if netMain is None:
    netMain = darknet.load_net_custom(configPath.encode(
        "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
if metaMain is None:
    metaMain = darknet.load_meta(metaPath.encode("ascii"))
if altNames is None:
    try:
        with open(metaPath) as metaFH:
            metaContents = metaFH.read()
            import re
            match = re.search("names *= *(.*)$", metaContents,
                                re.IGNORECASE | re.MULTILINE)
            if match:
                result = match.group(1)
            else:
                result = None
            try:
                if os.path.exists(result):
                    with open(result) as namesFH:
                        namesList = namesFH.read().strip().split("\n")
                        altNames = [x.strip() for x in namesList]
            except TypeError:
                pass
    except Exception:
        pass
darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
for file in glob.glob(path):
        #filename= path+ filename        
        frame_read = cv2.imread(file)
        
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                (darknet.network_width(netMain),
                                darknet.network_height(netMain)),
                                interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('Demo', image)
        cv2.waitKey()



cv2.destroyAllWindows()