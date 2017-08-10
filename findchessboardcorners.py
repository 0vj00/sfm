# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 09:47:49 2016

@author: venkat_reg
"""
import cv2
import numpy as np

NumCorners = (9,6)

objp = np.zeros((NumCorners[0]*NumCorners[1],3),np.float32)
objp[:,:2] = [[x,y] for y in range(NumCorners[1]) for x in range(NumCorners[0])]
objpoints=[]
imgpoints=[]

if 'cap' in globals():
    cap.release()
cv2.destroyAllWindows()

cap = cv2.VideoCapture(1)
assert( cap.isOpened() )

i = 0
pastSearchRes = False
mtx = None
while True:
    i = (i+1)%10
    ret, img = cap.read()
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if i == 0:
        ret, corners = cv2.findChessboardCorners(imggray, NumCorners, cv2.CALIB_CB_FAST_CHECK)
        pastSearchRes = ret
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(imggray, corners, (5,5), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
            print 'Length of objpoints is ', len(objpoints)
            
    if pastSearchRes == True:
        cv2.drawChessboardCorners(img, NumCorners, corners2, ret)
    

    if len(objpoints) > 30:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imggray.shape[::-1], None, None)
        print 'new mtx and dist are '        
        print mtx
        print dist

        h, w = imggray.shape
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        
        objpoints = []
        imgpoints = []
    
    if mtx is not None:
        undistimg = cv2.undistort(img, mtx, dist, None, newcameramtx)
        cv2.imshow('win2',undistimg)
        
    cv2.imshow('win1',img)
    
    
    if cv2.waitKey(66) & 0xFF == ord('q'):
        break
    


##objp = np.zeros(shape=(6*7,3),np.float32)
##objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
#
##objpoints = []
##imgpoints = []
#
#
##ret, img = cap.read()
#img = cv2.imread('./Sandbox1/OpenCV/opencv/samples/data/chessboard.png')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
#
#if ret == True:
#    #objpoints.append(objp)
#    
#    cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
#    #imgpoints.append(corners)
#    
#    cv2.drawChessboardCorners(img, (7,6), corners2, ret )
#    
#    cv2.imshow('img',img)
#    cv2.waitKey(500)


cap.release()
cv2.destroyAllWindows()