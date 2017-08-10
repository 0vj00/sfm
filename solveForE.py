# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 19:59:21 2016

@author: venkat_reg
"""
from __future__ import division
import cv2

def solveForE(pts1, pts2, mtxinv):
    
    def Solve(pts1HC3D, pts2HC3D):
        A = np.asarray([np.kron(pts1HC3D[i],pts2HC3D[i]) for i in  xrange(pts1HC3D.shape[0])])
        U, s, V = np.linalg.svd(A, full_matrices=True) # A = U.s.V
        condition, ratio = s[0]/s[-2], s[-1]/s[-2]
        print 'condition number %f, ratio %f'%(condition, ratio)
        #print s
        E = V[-1,:].reshape(3,3)
        UE, sE, VE = np.linalg.svd(E, full_matrices=True)
        #print 'svd est rank', estRank
        #print 'eigenvalues of E', sE
        print 'residual is', np.mean(np.abs(np.dot(pts1HC3D,np.dot(E,pts2HC3D.T))))
        #mid = np.zeros(A.shape)
        #mid[:A.shape[1], :A.shape[1]] = np.diag(s)
        E = np.dot(UE, np.dot(np.diag([1,1,0]),VE))
        pass
        return E, condition, ratio    
    
    def conditionThenSolve(pts1HC3D, pts2HC3D):
        pts1HC3DConditioned, T1Overall = conditionPts(pts1HC3D)        
        pts2HC3DConditioned, T2Overall = conditionPts(pts2HC3D)
        #print T1Overall
        #print T2Overall
        print np.mean(np.linalg.norm(pts1HC3DConditioned-pts2HC3DConditioned,axis=0))/np.max(np.abs(pts1HC3DConditioned))
        plotPts(pts1HC3DConditioned, pts2HC3DConditioned)
        E, condition, ratio = Solve(pts1HC3DConditioned, pts2HC3DConditioned)
        return np.dot(T1Overall.T, np.dot(E, T2Overall)), condition, ratio
        
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

    pts1HC3DConditioned, T1Overall = conditionPts(pts1HC3D)    
    pts2HC3DConditioned, T2Overall = conditionPts(pts2HC3D)
    print np.mean(pts1HC3D, axis=0)
    print np.mean(pts2HC3D, axis=0)
    plotPts(pts1HC3D, pts2HC3D)
    H,_ = cv2.findHomography(10*pts1HC3D[:,:2],10*pts2HC3D[:,:2],method=cv2.RANSAC)
    print H
    decH = cv2.decomposeHomographyMat(H,np.eye(3))
    print decH[0]
    print 'rotations'
    print decH[1]
    print 'translations'
    print decH[2]
    print 'normals'
    print decH[3]
#    lowestcondition = 100000
#    bestE = None
#    bestratio = None
#    for i in xrange(10000):
#        idx = np.random.randint(0,numpts, (8,))
#        E, condition, ratio = conditionThenSolve(pts1HC3D[idx,:], pts2HC3D[idx,:])
#        if condition < lowestcondition:
#            lowestcondition = condition
#            bestratio = ratio
#            bestE = E
#    print 'lowest condition %f'%lowestcondition
#    print 'best ratio %f'%bestratio
#    print bestE
#    Ue, se, Ve = np.linalg.svd(bestE)
#    W = np.array([[0, -1, 0],[1, 0, 0],[0, 0, 1]])
#    tx = np.dot(Ue, np.dot(W, np.dot(np.diag([1,1,0]), Ue.T)))
#    R = np.dot(Ue, np.dot(W.T, Ve))
#    
#    print R
#    print tx
#    # ransac. select N points to use; then do transform and then use he
#    
##    print np.mean(pts2HC3D-pts1HC3D, axis=0)
##    ##
##    ##
#    
    