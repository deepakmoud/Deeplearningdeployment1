# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 11:35:52 2021

@author: deepak
"""
import cv2
import math

cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
cascade_eye = cv2.CascadeClassifier('haarcascade_eye.xml') 
cascade_smile = cv2.CascadeClassifier('haarcascade_smile.xml')
imagesFolder = '/static/data-preprocess/new'
def detection(grayscale, img):
    face = cascade_face.detectMultiScale(grayscale, 1.3, 5)
    for (x_face, y_face, w_face, h_face) in face:
        cv2.rectangle(img, (x_face, y_face), (x_face+w_face, y_face+h_face), (255, 130, 0), 2)
        ri_grayscale = grayscale[y_face:y_face+h_face, x_face:x_face+w_face]
        ri_color = img[y_face:y_face+h_face, x_face:x_face+w_face] 
        eye = cascade_eye.detectMultiScale(ri_grayscale, 1.2, 18) 
        for (x_eye, y_eye, w_eye, h_eye) in eye:
            cv2.rectangle(ri_color,(x_eye, y_eye),(x_eye+w_eye, y_eye+h_eye), (0, 180, 60), 2) 
        smile = cascade_smile.detectMultiScale(ri_grayscale, 1.7, 20)
        for (x_smile, y_smile, w_smile, h_smile) in smile: 
            cv2.rectangle(ri_color,(x_smile, y_smile),(x_smile+w_smile, y_smile+h_smile), (255, 0, 130), 2)
    return img 

vc = cv2.VideoCapture(0) 

while True:
    ret, img = vc.read() 
    if (ret == True):
        frameId = vc.get(1) # current frame number
        print(frameId)
        frameRate = vc.get(5) # frame rate
        print(frameRate)
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        final = detection(grayscale, img) 
        cv2.imshow('Video', final)
        cv2.imwrite('/static/data-preprocess/new/image01.jpg', final)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
#filename = imagesFolder + "/image01" + ".jpg"
#print(filename)
#if (frameId % math.floor(frameRate) == 0):
 #   cv2.imwrite('/static/data-preprocess/new/image01.jpg', img)
 #   print(cv2.imwrite(filename, img))
vc.release() 
cv2.destroyAllWindows() 