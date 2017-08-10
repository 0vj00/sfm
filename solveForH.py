# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 19:59:21 2016

@author: venkat_reg
"""
from __future__ import division
import cv2
import numpy as np

def solveForH(pts1, pts2, mtxinv):
    def computeHandScore(pts1HC3DCond, pts2HC3DCond, HChoices, thresh):
        for i in xrange(3):
            H, mask = cv2.findHomography(pts1HC3DCond[HChoices,:2]/pts1HC3DCond[HChoices,2:3], 
                                        pts2HC3DCond[HChoices,:2]/pts2HC3DCond[HChoices,2:3],
                                        method = cv2.RANSAC, ransacReprojThreshold=1)
            if H is None or sum(mask) < 4:
                return None, None
            exppts2 = np.dot(H,pts1HC3DCond.T).T
            score = np.linalg.norm(exppts2[:,:2]/exppts2[:,2:3]-pts2HC3DCond[:,:2]/pts2HC3DCond[:,2:3],axis=1)
            HChoices = np.where(score<thresh)
            HChoices = HChoices[0]
            if len(HChoices) < 4:
                return None, None
        return H, score
    
    def conditionThenSolve(pts1HC3D, pts2HC3D):
        pts1HC3DConditioned, T1Overall = conditionPts(pts1HC3D)        
        pts2HC3DConditioned, T2Overall = conditionPts(pts2HC3D)
        #print T1Overall
        #print T2Overall
        #print np.mean(np.linalg.norm(pts1HC3DConditioned-pts2HC3DConditioned,axis=0))/np.max(np.abs(pts1HC3DConditioned))
        #plotPts(pts1HC3DConditioned, pts2HC3DConditioned)
        thresh = (1/600)*np.mean([T1Overall[0][0], T2Overall[0][0]])
        numpts = pts1HC3DConditioned.shape[0]
        ptsavail = np.ones((numpts,))
        Homographies_list = []
        for i in xrange(100):
            ptsselectable = np.where(ptsavail == 1)
            ptsselectable = ptsselectable[0]
            if ptsselectable.shape[0] < 4:
                break;
            tmp1 = np.random.randint(0,ptsselectable.shape[0], (4,))
            HChoices = ptsselectable[tmp1]
            H, HScore = computeHandScore(pts1HC3DConditioned, pts2HC3DConditioned, HChoices, thresh)
            if H is None:
                ptsavail[HChoices[0]] = 0
                continue
            ptsinH = np.where((HScore<thresh) & (ptsavail==1))
            ptsinH = ptsinH[0]
            H = np.dot(np.linalg.inv(T2Overall),np.dot(H,T1Overall))
            Homographies_list.append((H,ptsinH)) #, HScore))
            ptsavail[ptsinH] = 0
        return Homographies_list
        
    def conditionPts(ptsHC3D):
        meanPts = np.mean(ptsHC3D, axis=0)
        TShift = np.array([[1, 0, -meanPts[0]],
                           [0, 1, -meanPts[1]],
                            [0, 0, 1]])
        ptsHC3DShifted = np.dot(TShift,ptsHC3D.T).T
        scalePts = np.sqrt(2) / np.mean(np.linalg.norm(ptsHC3DShifted[:,:2], axis=1))
        TScale = np.array([[scalePts, 0, 0],
                           [0, scalePts, 0],
                            [0, 0, 1]])
        ptsHC3DConditioned = np.dot(TScale,ptsHC3DShifted.T).T
        TOverall = np.dot(TScale, TShift)
        return (ptsHC3DConditioned, TOverall)
    
    def plotPts(pts1, pts2):
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.scatter(pts1[:,0]/pts1[:,2], -pts1[:,1]/pts1[:,2])
        ax2.scatter(pts2[:,0]/pts2[:,2], -pts2[:,1]/pts2[:,2],edgecolors='red', marker='o', facecolors='none')
        plt.draw()
        plt.show()
        
    numpts = pts1.shape[0]
    pts1HC = np.hstack([pts1, np.ones(shape=(numpts,1))])
    pts2HC = np.hstack([pts2, np.ones(shape=(numpts,1))])
    pts1HC3D = np.dot(pts1HC, mtxinv.T)
    pts2HC3D = np.dot(pts2HC, mtxinv.T)
    Homographies_list = conditionThenSolve(pts1HC3D, pts2HC3D)
    # if there are too many homography planes, pick points from each plane and solve for E
    # otherwise, just solve for R and T from the homography. need to undo the R and T
    
#    for H,ptsinH in Homographies_list:
#        _, Rlist, Tlist, Nlist = cv2.decomposeHomographyMat(H, np.eye(3)) #mtx)
#        for R,T,N in zip(Rlist,Tlist,Nlist):
#            if R[2,2] > 0:
#                print 'rotation : ',R
#                print 'translation : ',T
#                print 'normal : ',N
#    
    return Homographies_list
    # ransac. select N points to use; then do transform and then use he
    
#    print np.mean(pts2HC3D-pts1HC3D, axis=0)
#    ##
#    ##
    
    