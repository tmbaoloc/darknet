import threading
import cv2
import darknet
class VideoCaptureThreading:
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.detections = []
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self, netMain, metaMain):
        if self.started:
            print('[!] Threaded video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(netMain, metaMain))
        self.thread.start()
        return self

    def update(self, netMain, metaMain):
        darknet_image = darknet.make_image(608,416,3)
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                if grabbed:
                    frame_resized = cv2.resize(frame,
                                (608,416), interpolation=cv2.INTER_LINEAR)
                    darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
                    self.detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
                    self.frame = frame_resized
                    self.grabbed = grabbed
                else:
                    self.grabbed = grabbed
                    self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame
    
    def read_m(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
            detections = self.detections.copy()
        return grabbed, frame, detections
    
    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()