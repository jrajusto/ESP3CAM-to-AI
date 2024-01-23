import cv2
import numpy as np
import requests
import time
import queue
import threading
import cvlib as cv
import matplotlib.pyplot as plt
from cvlib.object_detection import draw_bbox
 

 
URL = "http://192.168.1.67"
cap = cv2.VideoCapture(URL + ":81/stream")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if __name__ == '__main__':
    #requests.get(URL + "/control?var=framesize&val={}".format(8))
 
    while True:
        
        ret, frame = cap.read() 
        #Convert the image to grayscale
        gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #Detect faces in the images
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor = 1.3, minNeighbors=5, minSize=(100,100))
        #faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for(x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

        #Display the image with detected faces
        cv2.imshow('Image with Detected Faces',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
 
    cv2.destroyAllWindows()
    cap.release()