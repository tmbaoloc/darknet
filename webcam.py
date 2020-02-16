from ctypes import *
import math
import random
import os
import sys
import cv2
import numpy as np
import time
import darknet
from Capture import VideoCaptureThreading
import gspread
from oauth2client.service_account import ServiceAccountCredentials
# Ham sap xep contour tu trai sang phai tu tren xuong duoi

useDatabase = True
useGUI = False

def sort_contours(cnts):
    list1=[]
    list2=[]
    for c in cnts:
        x, y, w, h=cv2.boundingRect(c)        
        if y<50:
            list1.append(c)            
        else:
            list2.append(c)            
    sorted_list1 = sorted(list1, key=lambda ctr: cv2.boundingRect(ctr)[0])
    sorted_list2 = sorted(list2, key=lambda ctr: cv2.boundingRect(ctr)[0])
    cnts= sorted_list1 +sorted_list2 
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
    global pos
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    if useDatabase:
        if len(newString) > 6 and len(newString) < 9:
            sheet.update_cell(pos,1, newString)
            pos += 1
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
            #image1 = cv2.resize(image1, (300,200), 1)
            roi = image1
            gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,  85, 10)
            #kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            #thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
            #_, cont, _= cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            _, cont, _= cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            plate_info = ""
            for c in sort_contours(cont):
                (x, y, w, h) = cv2.boundingRect(c)
                ratio = h/w
                #if h < roi.shape[0]/2:
                if 1.5<=ratio<=3: # Chon cac contour dam bao ve ratio w/h
                    #if 0.3<=h/roi.shape[0]<=0.8: 
                    #if True:
                    if 0.25<=h/roi.shape[0]<=0.4: 
                        # Ve khung chu nhat quanh so
                        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Tach so va predict
                        curr_num = binary[y:y+h,x:x+w]
                        curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
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
            f_result = fine_tune(plate_info)
            if useGUI:
                cv2.imshow("Cac contour tim duoc", roi)
                cv2.putText(img,
                            " [" + f_result + "]",
                            (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            [25, 25, 225], 2)
            else:
                print(f_result)
            
        else: 
            continue
    return img



    

if __name__ == "__main__":
    start_time = time.time()

    ########## Database configuration ##########
    if useDatabase:
        scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',
                "https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name('Database.json', scope)
        client = gspread.authorize(creds)
        sheet = client.open('Database_ANPR').sheet1

    ########### Variables declaration ##########
    global model_svm, metaMain, netMain, altNames, digit_w, digit_h, pos
    netMain = None
    metaMain = None
    altNames = None
    pos = 2
    model_svm = cv2.ml.SVM_load('svm.xml')
    digit_w = 30 # Kich thuoc ki tu
    digit_h = 60 # Kich thuoc ki tu
    configPath = "./LP/yolov3-tiny_obj.cfg"
    weightPath = "./yolov3-tiny_obj_4000.weights"
    metaPath = "./LP/LP.data"

    ########## Path validation ##########
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


    cap = VideoCaptureThreading(0)
    cap.start()
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    print("Starting the YOLO loop...")
    
    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    init_time = time.time() - start_time
    print("Start time: ",init_time)
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        if ret:
            darknet.copy_image_from_bytes(darknet_image,frame_read.tobytes())
            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
            image = cvDrawBoxes(detections, frame_read)
            fps=str(int((1/(time.time()-prev_time))))
            if useGUI:
                cv2.putText(image,
                            "FPS: "+ fps,
                            (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            [25, 25, 225], 2)
                #print("FPS : %0.1f" %fps)
                cv2.imshow('Demo', image)
            else:
                print(fps)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.stop()
    cv2.destroyAllWindows()
    run_time = time.time() - start_time
    print("Run time: ",run_time)

