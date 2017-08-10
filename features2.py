# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 01:44:40 2016

@author: venkat_reg
"""
from __future__ import division
import cv2
import numpy as np
import cPickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def queryCamProperties(cap):
    print 'width %d'%cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print 'height %d'%cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
def ClearAndResize(ax, lims):
     ax.cla()
     ax.set_xlim3d(*lims[0])
     ax.set_ylim3d(*lims[1])
     ax.set_zlim3d(*lims[2])
     
def PlotQuiver(ax, mat, l):
    assert(mat.shape == (3,6))
    c = ['red', 'green', 'black']
    for i in range(3):
        ax.quiver3D(*mat[i], pivot = 'tail', length = l, color = c[i])
    plt.draw()

def PlotPtCloud(ax, ptcld):
    print 'ptcld shape',ptcld.shape
    assert(ptcld.shape[0] == 3)
    ax.scatter(ptcld[0,:],ptcld[1,:],ptcld[2,:])
    plt.draw()

def normalizePhotogrammetricModel(bfmatcher, ref, current):
    refCld, refDes = ref
    curCld, curDes = current
    
    matches = bfmatcher.match(refDes, curDes)
    print 'in normalizePhotogrammetricModel; found %d matches'%len(matches)
    matchQuality = np.asarray([m.distance for m in matches])
    curIdx = np.asarray([m.trainIdx for m in matches])
    refIdx = np.asarray([m.queryIdx for m in matches])
    
    reorder = matchQuality.argsort() # can truncate this if needed
    curIdx = curIdx[reorder]
    refIdx = refIdx[reorder]
    matchQuality = matchQuality[reorder]
    nummatches = len(matchQuality)

    curCld = curCld[:,curIdx]
    refCld = refCld[:,refIdx]
 
    ptarr = np.random.randint(0,nummatches,size=(CONF_samples_photogram_ratio,2))
    curDist = (curCld[:,ptarr[:,1]] - curCld[:,ptarr[:,0]])
    refDist = (refCld[:,ptarr[:,1]] - refCld[:,ptarr[:,0]])
    mask = refDist > 0.0
    curDist = curDist[mask]
    refDist = refDist[mask]
    ratios = np.linalg.norm(curDist,axis=0) / np.linalg.norm(refDist,axis=0)
    ratios = ratios[~np.isnan(ratios)]
    return np.median(ratios)

def trackFeatures(img1, img2, pts1):
    pts2, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, pts1, None, \
                                winSize = (21,21), \
                                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001))
    status = status.reshape(-1)
    pts2 = pts2[(status > 0),:]
    pts1 = pts1[(status > 0),:]
    return pts1, pts2

def solveForE(pts1, pts2, mtxinv, mtx2inv = None):
    if mtx2inv is None:
        mtx2inv = mtxinv
    numpts = pts1.shape[0]
    pts1HC = np.hstack([pts1, np.ones(shape=(numpts,1))])
    pts2HC = np.hstack([pts2, np.ones(shape=(numpts,1))])
    pts1HC3D = np.dot(pts1HC, mtxinv.T)
    pts2HC3D = np.dot(pts2HC, mtx2inv.T)
#    ##
#    fig2 = plt.figure()
#    ax2 = fig2.add_subplot(111)
#    ax2.scatter(pts1HC3D[:,0]/pts1HC3D[:,2], -pts1HC3D[:,1]/pts1HC3D[:,2])
#    ax2.scatter(pts2HC3D[:,0]/pts2HC3D[:,2], -pts2HC3D[:,1]/pts2HC3D[:,2],edgecolors='red', marker='o', facecolors='none')
#    plt.draw()
#    plt.show()
#    ##
    A = np.asarray([np.kron(pts1HC3D[i],pts2HC3D[i]) for i in  xrange(pts1HC3D.shape[0])])
    U, s, V = np.linalg.svd(A, full_matrices=True) # A = U.s.V
    estRank = sum( s/s[-1] >= 10 )
    E = V[-1,:].reshape(3,3)
    UE, sE, VE = np.linalg.svd(E, full_matrices=True)
    #print 'svd est rank', estRank
    #print 'eigenvalues of E', sE
    #print 'residual is', np.dot(pts1HC3D,np.dot(E,pts2HC3D.T))
    #mid = np.zeros(A.shape)
    #mid[:A.shape[1], :A.shape[1]] = np.diag(s)
    E = np.dot(UE, np.dot(np.diag([1,1,0]),VE))
    pass
    return E
    
def computeResidual(E, pts1, pts2, mtxinv):
    F = np.dot(mtxinv.T, np.dot(E, mtxinv))
    numpts = pts1.shape[0]
    pts1HC = np.hstack([pts1, np.ones(shape=(numpts,1))])
    pts2HC = np.hstack([pts2, np.ones(shape=(numpts,1))])
    return np.mean(np.dot( pts1HC, np.dot(F, pts2HC.T)))
    
if 'cap' in globals():
    cap.release()
cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
assert( cap.isOpened() )
# if cam2
if 'cap2' in globals():
    cap2.release()
cap2 = cv2.VideoCapture(1)
assert( cap2.isOpened() )

plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

mtx, dist = cPickle.load(open('microsoftHD3Kcalib.mtxdist','rb'))
mtxinv = np.linalg.inv( mtx )
# if cam2
mtx2, dist2 = cPickle.load(open('HPcalib.mtxdist','rb'))
mtx2inv = np.linalg.inv( mtx2 )

ffd = cv2.FastFeatureDetector_create(nonmaxSuppression = True, threshold = 20)

coord_world = [np.zeros(shape=(3,3)), np.eye(3)]
prevImg = None
prevFeaturePts = None
matchedprevFeaturePts = None

####### doing sanity check on E calculations...
#pts_wrt_world = np.array([[0],[0],[10],[0]]) + np.array( [[0,0,0,1],[1,0,0,1],[0,1,0,1],[1,1,0,1],
#                 [0,0,1,1],[1,0,1,1],[0,1,1,1],[1,1,1,1],
#                 [2,1,1,1],[1,2,1,1],[2,2,2,1],[2,1,2,1],
#                 [3,1,2,1],[3,2,2,1],[3,3,1,1,],[2,1,3,1]] ).T
#
#projmatcam1 = np.dot(mtx, np.hstack([np.eye(3),np.zeros(shape=(3,1))]))
#R = np.eye(3)
#T = np.array([1,0,0])[np.newaxis].T
#Tx = np.array([[0,-0,0],[0,0,-1],[-0,1,0]])
#
#projmatcam2 = np.dot(mtx, np.hstack([R.T,np.dot(-R.T,T)]))
#pts_cam1 = np.dot(projmatcam1,pts_wrt_world)
#pts_cam2 = np.dot(projmatcam2, pts_wrt_world)
#pts_cam1 = pts_cam1/pts_cam1[-1,:]
#pts_cam2 = pts_cam2/pts_cam2[-1,:]
#pts_cam1 = pts_cam1[:2,:].T
#pts_cam2 = pts_cam2[:2,:].T
#
#expectedE = np.dot(Tx,R)
#E = solveForE(pts_cam1, pts_cam2, mtxinv)
#E2 = cv2.findEssentialMat(pts_cam1, pts_cam2, mtx, method = cv2.RANSAC, threshold=1)
#print expectedE
#print E
#print E2[0]
#cap.release()
#cap2.release()
#cv2.destroyAllWindows()
## conclusion: with only 8 cube points, there wasn't enough information to determine E propoerly. Adding more points helps to get a good estimate of E.
## projection of the 2d points to the 3d coordinates as seen by each camera has been verified to be accurate
####### end sanity....

leftcamframe = cPickle.load(open('leftcam.frm','rb'))
rightcamframe = cPickle.load(open('rightcam.frm','rb'))

while True:
    #ret, img = cap.read()
    img = leftcamframe.copy()
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # if cam2
    #ret2, img2 = cap2.read()
    img2 = rightcamframe.copy()
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)    

    kp = ffd.detect(imggray, None)
    cam1FeaturePts = np.asarray([k.pt for k in kp]).astype('float32')
    
    matchedcam1FeaturePts, matchedcam2FeaturePts = trackFeatures(imggray, img2gray, cam1FeaturePts)
    
    
    matchedkp = [cv2.KeyPoint(x,y,3) for x,y in matchedcam1FeaturePts]
    matchedkp2 = [cv2.KeyPoint(x,y,3) for x,y in matchedcam2FeaturePts]
    
    matchedFeatureSetSize = matchedcam2FeaturePts.shape[0]
    print 'FeatureSetSize is %d'%matchedFeatureSetSize
    
    if matchedFeatureSetSize > 5:
            E = solveForE(matchedcam1FeaturePts, matchedcam2FeaturePts, mtxinv)
            E2, _ = cv2.findEssentialMat(matchedcam1FeaturePts, matchedcam2FeaturePts, mtx, method = cv2.RANSAC, threshold=1)
#            #residual = computeResidual(E, matchedprevFeaturePts, currFeaturePts, mtxinv)
#            ##pts_img1_corrected, pts_img2_corrected = cv2.correctMatches()
#            cam2ptsontocam1 = np.dot(mtx, np.dot(mtx2inv, 
#                                np.hstack([matchedcam2FeaturePts, np.ones(shape=(matchedcam2FeaturePts.shape[0],1))]).T))
#            cam2ptsontocam1 = (cam2ptsontocam1[:2,:]/cam2ptsontocam1[-1,:]).T
            _, R, T, mask = cv2.recoverPose(E, matchedcam1FeaturePts, matchedcam2FeaturePts, mtx)
            _, R2, T2, mask2 = cv2.recoverPose(E2, matchedcam2FeaturePts, matchedcam2FeaturePts, mtx)
            
            
#            coord_cam_prev = coord_world
#            coord_cam_curr = [ coord_cam_prev[0] + np.dot(np.dot(coord_cam_prev[1],T), np.ones(shape=(1,3))),
#                                  np.dot(coord_cam_prev[1], R)]
#            
#            ClearAndResize(ax, ((-50,50),(-50,50),(0,50)))
#            PlotQuiver(ax, np.vstack(coord_cam_prev).T, 10)
#            PlotQuiver(ax, np.vstack(coord_cam_curr).T, 10)
            pass
#    
    img = cv2.drawKeypoints(img, matchedkp, None, color=(0,0,255))
    img2 = cv2.drawKeypoints(img2, matchedkp2, None, color=(0,0,255))
    if matchedcam1FeaturePts is not None:
        for i in xrange(matchedcam1FeaturePts.shape[0]):
            img = cv2.line(img, tuple(matchedcam1FeaturePts[i,:]), tuple(matchedcam2FeaturePts[i,:]),color=(0,255,0) )
            img2 = cv2.line(img2, tuple(matchedcam2FeaturePts[i,:]), tuple(matchedcam1FeaturePts[i,:]),color=(0,255,0) )
    img3 = np.hstack([img, img2])
    cv2.imshow('win1', img3)
    if cv2.waitKey(66) & 0xFF == ord('q'):
        break
    # to do : add new pts to here...
    
    

#coord_world = [np.zeros(shape=(3,3)), np.eye(3)]
#prevImg = None
#prevFeaturePts = None
#matchedprevFeaturePts = None
#while True:
#    ret, img = cap.read()
#    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    
#    if prevImg is None:
#        kp = ffd.detect(imggray, None)
#        currFeaturePts = np.asarray([k.pt for k in kp]).astype('float32')
#    else:
#        matchedprevFeaturePts, currFeaturePts = trackFeatures(prevImg, imggray, prevFeaturePts)
#        kp = [cv2.KeyPoint(x,y,7) for x,y in currFeaturePts]
#        matchedFeatureSetSize = currFeaturePts.shape[0]
#        print 'FeatureSetSize is %d'%matchedFeatureSetSize
#    
#        if matchedFeatureSetSize > 5:
#            solveForE(matchedprevFeaturePts, currFeaturePts, mtxinv)
#            E, _ = cv2.findEssentialMat(matchedprevFeaturePts, currFeaturePts, mtx, method = cv2.RANSAC, threshold=1)
#            residual = computeResidual(E, matchedprevFeaturePts, currFeaturePts, mtxinv)
#            #pts_img1_corrected, pts_img2_corrected = cv2.correctMatches()
#            _ , R, T, mask = cv2.recoverPose(E, matchedprevFeaturePts, currFeaturePts, mtx)    
#    
#            coord_cam_prev = coord_world
#            coord_cam_curr = [ coord_cam_prev[0] + np.dot(np.dot(coord_cam_prev[1],T), np.ones(shape=(1,3))),
#                                  np.dot(coord_cam_prev[1], R)]
#        
#            print residual
#            # axis_cam and loc_cam represent the rotation and translation of the camera axis w.r.t. world. 
#            proj_cam_prev = np.dot(mtx, \
#                np.hstack( [coord_cam_prev[1].T, np.dot(-coord_cam_prev[1].T, coord_cam_prev[0][:,0][np.newaxis].T)] ) )
#            proj_cam_curr  = np.dot(mtx, \
#                np.hstack( [coord_cam_curr[1].T, np.dot(-coord_cam_curr[1].T, coord_cam_curr[0][:,0][np.newaxis].T)] ) )
#            featureCloud  = cv2.triangulatePoints(proj_cam_prev, proj_cam_curr, matchedprevFeaturePts.T, currFeaturePts.T)
#            featureCloud  = featureCloud/featureCloud[-1,:] # from HC to XYZ
#            
#            ClearAndResize(ax, ((-50,50),(-50,50),(0,50)))
#            PlotQuiver(ax, np.vstack(coord_cam_prev).T, 10)
#            PlotQuiver(ax, np.vstack(coord_cam_curr).T, 10)
#            #PlotPtCloud(ax, featureCloud)
#                
#            
#    
#    img = cv2.drawKeypoints(img, kp, None, color=(0,0,255))
#    if matchedprevFeaturePts is not None:
#        for i in xrange(matchedprevFeaturePts.shape[0]):
#            img = cv2.line(img, tuple(matchedprevFeaturePts[i,:]), tuple(currFeaturePts[i,:]),color=(0,255,0) )
#    cv2.imshow('win1', img)
#    if cv2.waitKey(66) & 0xFF == ord('q'):
#        break
#    # to do : add new pts to here...
#    prevImg = imggray
#    if currFeaturePts.shape[0] < 300:
#        kp = ffd.detect(imggray, None)
#        if len(kp) > 0:
#            additionalFeaturePts = np.asarray([k.pt for k in kp]).astype('float32')
#            currFeaturePts = [tuple(elem) for elem in np.vstack([currFeaturePts, additionalFeaturePts])]
#            currFeaturePts = np.asarray(currFeaturePts)
#        
#    prevFeaturePts = currFeaturePts    
cap.release()
cap2.release()
cv2.destroyAllWindows()