# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 09:47:49 2016

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
fig = plt.figure()
fig2 = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax2 = fig2.add_subplot(111,projection='3d')


NumCorners = (9,6)

objp = np.zeros((NumCorners[0]*NumCorners[1],3),np.float32)
objp[:,:2] = [[1*x,1*y] for y in range(NumCorners[1]) for x in range(NumCorners[0])]

mtx, dist = cPickle.load(open('microsoftHD3Kcalib.mtxdist','rb'))

if 'cap' in globals():
    cap.release()
cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
assert( cap.isOpened() )

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
axis = np.float32([[3,0,0], 
                   [0,3,0],
                   [0,0,3]])

i=0;                   
while True:
    ret, img = cap.read()
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(imggray, NumCorners, cv2.CALIB_CB_FAST_CHECK)
    if ret == True:
        corners2 = cv2.cornerSubPix(imggray, corners, (5,5), (-1,-1), criteria)
        ret, rvecs, tvecs = cv2.solvePnP(objp.reshape(-1,1,3), corners2, mtx, dist)
        #ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp.reshape(-1,1,3), corners2, mtx, dist)
        if i%10 == 0:
           drawRvecsTvecs(ax, rvecs, tvecs)
           R,_ = cv2.Rodrigues(rvecs)
           axis_cam = np.eye(3)
           axis_obj = np.dot(R,axis_cam)
           ClearAndResize(ax2,((-50,50),(-50,50),(0,50)))
           
           PlotQuiver(ax2,np.vstack([np.zeros(shape=(3,3)), axis_cam]).T, 10)
           PlotQuiver(ax2,np.vstack([np.dot(tvecs,np.ones(shape=(1,3))), axis_obj]).T, 10)
           
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        drawAxes(img, corners2, imgpts)
    
    
    #undistimg = cv2.undistort(img, mtx, dist, None, newcameramtx)
    #cv2.imshow('win2',undistimg)
        
    cv2.imshow('win1',img)
    
    
    if cv2.waitKey(66) & 0xFF == ord('q'):
        break
    
plt.close('all')
cap.release()
cv2.destroyAllWindows()

