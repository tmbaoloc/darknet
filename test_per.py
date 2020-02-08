from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
#import pytesseract


def sort_contours(cnts):

    reverse = False
    i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax
char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ.-'

# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
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
        #cv2.imshow('test', image1)
        if image1.size !=0:
            digit_w = 30 # Kich thuoc ki tu
            digit_h = 60 # Kich thuoc ki tu
            roi = image1
            
            #continue
            gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            #binary=cv2.threshold(gray, 127, 255,
                         #cv2.THRESH_BINARY_INV)[1]
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115, 1)
            kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
            _, cont, _= cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #cv2.imshow("Cac contour tim duoc", binary)

            plate_info = ""

            for c in sort_contours(cont):
                (x, y, w, h) = cv2.boundingRect(c)
                ratio = h/w
                if 1.5<=ratio<=3.5: # Chon cac contour dam bao ve ratio w/h
                    if 0.3<=h/roi.shape[0]<=0.8: 

                        # Ve khung chu nhat quanh so
                        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Tach so va predict
                        curr_num = thre_mor[y:y+h,x:x+w]
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
            #print("Bien so=", plate_info)
            #cv2.imshow("Cac contour tim duoc", roi)
            # gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            # black = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
            # resz = cv2.resize(black, (150,100), 1)
            # #cv2.imshow('crop', black)
            # custom_tess= r'--oem 3 --psm 11'    
            # ricacdo = pytesseract.image_to_string(resz,lang='eng',config=custom_tess)
            # print('Bien so: ',fine_tune(ricacdo))
            cv2.putText(img,
                        " [" + fine_tune(plate_info) + "]",
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [25, 25, 225], 2)
            
        else: 
            continue
    return img

def cropimg(detections, image):
    crop = image.copy()
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        image = crop[ymin:ymax,xmin:xmax]
        
        #blur = cv2.GaussianBlur(grayscaled, (5,5),0)
        #cac = cv2.threshold(grayscaled, 127, 255, cv2.THRESH_BINARY |cv2.THRESH_OTSU)
        #scale = cv2.resize(black, (200,150), 1)
        #a=str(round(detection[1] * 100, 2)) //hien thi phan tram 
        #print(a)
        
    return image
netMain = None
metaMain = None
altNames = None


def YOLO():
    start_time = time.time()
    global model_svm
    model_svm = cv2.ml.SVM_load('svm.xml')
    global metaMain, netMain, altNames
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
    
    cap = cv2.VideoCapture(0)
    
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    init_time = time.time() - start_time
    print("Start time: %0.3f",init_time)
    # while True:
    #     prev_time = time.time()
    #     ret, frame_read = cap.read()
    #     if frame_read.size !=0:
    #         frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
    #         frame_resized = cv2.resize(frame_rgb,
    #                                    (darknet.network_width(netMain),
    #                                     darknet.network_height(netMain)),
    #                                    interpolation=cv2.INTER_LINEAR)
    #         darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
    #         detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
    #         image = cvDrawBoxes(detections, frame_resized)
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     #bienso = cropimg(detections, frame_resized)
    #     #gray = cv2.cvtColor(bienso, cv2.COLOR_BGR2GRAY)
    #     #black = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
        
    #     #custom_tess= r'--oem 3 --psm 6'    
    #     #ricacdo = pytesseract.image_to_string(black,lang=None,config=custom_tess)
    #     #print('Bien so: ',fine_tune(ricacdo))
    #     #cv2.imshow('crop', bienso)
    #         fps=(1/(time.time()-prev_time))+1
    #         print("FPS : %0.1f" %fps)
    #         cv2.imshow('Demo', image)
        
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    cap.release()
    cv2.destroyAllWindows()
    # run_time = time.time() - start_time
    # print("Run time: ",run_time)
    

if __name__ == "__main__":
    YOLO()

