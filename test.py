#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : linjie
import cv2 as cv
#读取视频信息。
cap = cv.VideoCapture("http://admin:admin@172.20.24.19:8081")  #@前为账号密码，后为ip地址
face_xml = cv.CascadeClassifier("haarcascade_frontalface_default.xml") #导入XML文件
while(cap.isOpened()):
    f,img = cap.read()   #读取一帧图片
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)  #转换为灰度图
    face = face_xml.detectMultiScale(gray,1.3,10)    #检测人脸，并返回人脸位置信息

    for (x,y,w,h) in face:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv.imshow("1",img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
