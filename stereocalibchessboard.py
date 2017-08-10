# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 20:35:38 2016

@author: venkat_reg
"""

import cv2
import numpy as np
import cPickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def drawAxes(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 3)

def drawRvecsTvecs(ax, rvecs, tvecs):
     ClearAndResize(ax,((-1,1),(-1,1),(-1,1)))
     ax.quiver(*(np.vstack([np.zeros(shape=(3,1)), rvecs])), pivot = 'tail', color = 'red')
     ax.quiver(*(np.vstack([np.zeros(shape=(3,1)), tvecs])), pivot = 'tail', color = 'blue')        
     plt.draw()

def ClearAndResize(ax, lims):
     ax.cla()
     ax.set_xlim3d(*lims[0])
     ax.set_ylim3d(*lims[1])
     ax.set_zlim3d(*lims[2])
     
def PlotQuiver(ax, mat, l):
    assert(mat.shape == (3,6))
    # change x->x; y-> -1*z; z->y
    #mat = np.dot(np.array([[1,0,0],
     #                      [0,0,-1],
      #                     [0,1,0]]), mat)
    c = ['red', 'green', 'black']
    for i in range(3):
        ax.quiver3D(*mat[i], pivot = 'tail', length = l, color = c[i])
    plt.draw()

plt.close('all')
#fig = plt.figure()
fig2 = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
ax2 = fig2.add_subplot(111,projection='3d')


NumCorners = (9,6)

objp = np.zeros((NumCorners[0]*NumCorners[1],3),np.float32)
objp[:,:2] = [[1*x,1*y] for y in range(NumCorners[1]) for x in range(NumCorners[0])]
objpts = []
img1pts = []
img2pts = []

mtx1, dist1 = cPickle.load(open('microsoftHD3Kcalib.mtxdist','rb'))
mtx2, dist2 = cPickle.load(open('HPcalib.mtxdist','rb'))
if 'cap1' in globals():
    cap1.release()
if 'cap2' in globals():
    cap2.release()
cv2.destroyAllWindows()

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
assert( cap1.isOpened() and cap2.isOpened() )

drawchessboard = False

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
axis = np.float32([[3,0,0], 
                   [0,3,0],
                   [0,0,3]])

i=0;    
RO_known = False               
while True:
    i = (i+1)%10
    
    ret1, img1 = cap1.read()
    ret2, img2 = cap2.read()
    img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)    
    
    if 1==1: #RO_known == False:
        if i%3 == 0:
            ret1, corners1 = cv2.findChessboardCorners(img1gray, NumCorners, cv2.CALIB_CB_FAST_CHECK)
            ret2, corners2 = cv2.findChessboardCorners(img2gray, NumCorners, cv2.CALIB_CB_FAST_CHECK)    
            drawchessboard = ret1 and ret2        
            if ret1 == True and ret2 == True:
                objpts.append(objp)
                corners1subpix = cv2.cornerSubPix(img1gray, corners1, (5,5), (-1,-1), criteria)
                corners2subpix = cv2.cornerSubPix(img2gray, corners2, (5,5), (-1,-1), criteria)
                img1pts.append(corners1subpix)
                img2pts.append(corners2subpix)
                print 'length of objpoints is %d'%len(objpts)
            
        if drawchessboard == True:
            cv2.drawChessboardCorners(img1, NumCorners, corners1subpix, ret1)
            cv2.drawChessboardCorners(img2, NumCorners, corners2subpix, ret2)
        
        if len(objpts) > 30:
            assert(img1gray.shape == img2gray.shape)
            ret, mtx1new, dist1new, mtx2new, dist2new, R, T, E, F = \
            cv2.stereoCalibrate(objpts, img1pts, img2pts, mtx1, dist1, mtx2, dist2, img1gray.shape[::-1], \
                        flags = cv2.CALIB_FIX_INTRINSIC, criteria = criteria)
            RO_known = True
            objpts  = []
            img1pts = []
            img2pts = []
            # T seems to be translation of first w.r.t second.
            
            
    if RO_known == True and i==0:
        ret1, corners1 = cv2.findChessboardCorners(img1gray, NumCorners, cv2.CALIB_CB_FAST_CHECK)
        ret2, corners2 = cv2.findChessboardCorners(img2gray, NumCorners, cv2.CALIB_CB_FAST_CHECK) 
        if ret1 and ret2:        
            corners1subpix = cv2.cornerSubPix(img1gray, corners1, (5,5), (-1,-1), criteria)
            corners2subpix = cv2.cornerSubPix(img2gray, corners2, (5,5), (-1,-1), criteria)
            ret, rvecs1, tvecs1 = cv2.solvePnP(objp.reshape(-1,1,3), corners1subpix, mtx1, dist1)
            ret, rvecs2, tvecs2 = cv2.solvePnP(objp.reshape(-1,1,3), corners2subpix, mtx2, dist2)        
            R1,_ = cv2.Rodrigues(rvecs1)
            R2,_ = cv2.Rodrigues(rvecs2)
        axis_cam1 = np.eye(3)
        loc_cam1  = np.zeros(shape=(3,3))
        axis_cam2 = np.dot(R.T,axis_cam1)
        loc_cam2  = np.dot(np.dot(-R.T,T),np.ones((1,3)))
        
        ClearAndResize(ax2,((-50,50),(-50,50),(0,150)))
           
        PlotQuiver(ax2,np.vstack([loc_cam1, axis_cam1]).T, 10)
        PlotQuiver(ax2,np.vstack([loc_cam2, axis_cam2]).T, 10)
        
        if ret1 and ret2:
           axis_obj1 = np.dot(R1,axis_cam1)
           axis_obj2 = np.dot(R2,axis_cam2)
           loc_obj1  = loc_cam1 + np.dot(tvecs1,np.ones(shape=(1,3)))
           loc_obj2  = loc_cam2 + np.dot(np.dot(R.T,tvecs2),np.ones(shape=(1,3)))
           PlotQuiver(ax2,np.vstack([loc_obj1, axis_obj1]).T, 10)
           PlotQuiver(ax2,np.vstack([loc_obj2, axis_obj2]).T, 10)
           
           img1ptsaxis, _ = cv2.projectPoints(axis, rvecs1, tvecs1, mtx1, dist1)
           img2ptsaxis, _ = cv2.projectPoints(axis, rvecs2, tvecs2, mtx2, dist2) 
           drawAxes(img1, corners1subpix, img1ptsaxis)
           drawAxes(img2, corners2subpix , img2ptsaxis)
           
    
    
    #undistimg = cv2.undistort(img, mtx, dist, None, newcameramtx)
    #cv2.imshow('win2',undistimg)
        
    cv2.imshow('win1',img1)
    cv2.imshow('win2',img2)
    
    if cv2.waitKey(66) & 0xFF == ord('q'):
        break
    
plt.close('all')
cap.release()
cv2.destroyAllWindows()

