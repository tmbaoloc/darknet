from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import pytesseract

os.environ['OMP_THREAD_LIMIT'] = '4'
def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax
char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ.-'

def sort_contours(cnts):

    reverse = False
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts
# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString


def get_best_images(input_frames):
     # Sap xep cac input_frames theo muc do Focus giam dan
     input_frames = sorted(input_frames, key=lambda img : cv2.Laplacian(img[0], cv2.CV_64F).var(), reverse=True)
     # Lay khung hinh co muc do focus tot nhat
     best_image = input_frames[:1]
     return best_image


def cvDrawBoxes(detections, img):
    global frame_count
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
        
        #if frame_count % 5 == 0:
            #bestimg = get_best_images(image1)
        if image1.size !=0:
                #print(frame_count)
               
                #cv2.imshow('crop', image1)
                gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                black = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
                rez = cv2.resize(black, (150,100), 1)
                #cv2.imshow('crop', rez)
                if frame_count % 5 == 0:
                    custom_tess= r'--oem 3 --psm 12 -c tessedit_do_invert=0'    
                    ricacdo = pytesseract.image_to_string(rez,lang='eng',config=custom_tess)
                    print('Bien so: ',fine_tune(ricacdo))
                    cv2.putText(img,
                                detection[0].decode() +
                                " [" + fine_tune(ricacdo) + "]",
                                (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                [25, 25, 225], 2)
                frame_count += 1                
        else: 
            continue
        #frame_count += 1
    return img

frame_count = 0
netMain = None
metaMain = None
altNames = None


def YOLO():

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
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        if frame_read.size !=0:
            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb,
                                       (darknet.network_width(netMain),
                                        darknet.network_height(netMain)),
                                       interpolation=cv2.INTER_LINEAR)
            darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
            image = cvDrawBoxes(detections, frame_resized)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #bienso = cropimg(detections, frame_resized)
        #gray = cv2.cvtColor(bienso, cv2.COLOR_BGR2GRAY)
        #black = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
        
        #custom_tess= r'--oem 3 --psm 6'    
        #ricacdo = pytesseract.image_to_string(black,lang=None,config=custom_tess)
        #print('Bien so: ',fine_tune(ricacdo))
        #cv2.imshow('crop', bienso)
            fps=1/(time.time()-prev_time)
            print("FPS : %0.1f" %fps)
            cv2.imshow('Demo', image)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    YOLO()

