# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 09:45:21 2016

@author: venkat_reg
"""
from __future__ import division
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
#    for i in xrange(10000):
#        pt1,pt2 = np.random.randint(0, nummatches,size=(2,))
#        if pt1 == pt2:
#            continue
#        curDist = np.linalg.norm(curCld[:,pt2] - curCld[:,pt1])
#        refDist = np.linalg.norm(refCld[:,pt2] - refCld[:,pt1])
#        thisratio = curDist / refDist
#        if np.isnan(thisratio) or thisratio == 0:
#            continue
#        ratios.append(np.log( thisratio ))
#    fig4 = plt.figure()
#    ax4 = fig4.add_subplot(111)
#    ax4.hist(ratios, bins = 100)
#    curCld = curCld - curCld[:,0][np.newaxis].T
#    refCld = refCld - refCld[:,0][np.newaxis].T
#    curCld = np.sqrt(curCld[0,:]**2 + curCld[1,:]**2 + curCld[2,:]**2)
#    refCld = np.sqrt(refCld[0,:]**2 + refCld[1,:]**2 + refCld[2,:]**2)
#    fig4 = plt.figure()
#    ax4 = fig4.add_subplot(111)
#    ax4.plot(curCld[1:]/refCld[1:])
    return np.median(ratios)

plt.close('all')
fig = plt.figure()
fig2 = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax2 = fig2.add_subplot(111,projection='3d')

# Configuration
CONF_keypts_hist_len = 1
CONF_campts_hist_len = 1
CONF_corr_closeness = 1
CONF_samples_photogram_ratio = 10000

if 'cap' in globals():
    cap.release()
cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
assert( cap.isOpened() )

mtx, dist = cPickle.load(open('microsoftHD3Kcalib.mtxdist','rb'))

orb = cv2.ORB_create(nfeatures=50)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
i = 0

keypts_hist=[] # stores (keypoints, description, image)
campts_hist=[] # stores (location, axis, rotation_matrix), all are w.r.t world coordinates
referencePhotogrammetricModel = None

# need world to be origin and unit coords...not sure code will work if this is changed
axis_world = np.eye(3)
loc_world  = np.zeros(shape=(3,3))

campts_hist.append((loc_world, axis_world))
while True:
    i = (i+1)%10
    
    ret, img = cap.read()
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    kp, des = orb.detectAndCompute(imggray,None)    
    #print 'found %d key points'%(len(kp))

    if des is not None and i == 0:
        # store keypoints and image
        keypts_hist.append((kp,des,img))
        if len(keypts_hist) > CONF_keypts_hist_len:
            keypts_hist.pop(0)

    img = cv2.drawKeypoints(img, kp, img, color=(0,255,0), flags = 0)
    
    if des is not None and len(keypts_hist) > 0:
        # try to match keypoints with those from past
        assert(keypts_hist[0][1].dtype == des.dtype and keypts_hist[0][1].shape[1] == des.shape[1])
        
        matches = bf.match(keypts_hist[0][1], des)
        #print 'found %d matches; %f, %f percent'%(len(matches), 100*len(matches)/keypts_hist[0][1].shape[0], 100*len(matches)/des.shape[0])
        matched_kp = [kp[m.trainIdx] for m in matches]
        matched_kp_in_hist = [keypts_hist[0][0][m.queryIdx] for m in matches]
        matched_des = np.asarray([des[m.trainIdx] for m in matches])
        
        img = cv2.drawKeypoints(img, matched_kp, img, color=(0,0,255), flags = 0)
        img_hist = keypts_hist[0][2]
        img_hist = cv2.drawKeypoints(img_hist, keypts_hist[0][0], img_hist, color=(0,255,0), flags=0)
        img_hist = cv2.drawKeypoints(img_hist, matched_kp_in_hist, img_hist, color=(0,0,255), flags=0)
        
        cv2.imshow('winhist',img_hist)
        
        if len(matched_kp) > 10:
        
            pts_img1 = np.asarray([k.pt for k in matched_kp_in_hist])
            pts_img2 = np.asarray([k.pt for k in matched_kp])
            correspondence_closeness = np.abs(pts_img1-pts_img2).sum() / pts_img1.shape[0]
            print 'corrspondence pts closeness %d pix per pt'%correspondence_closeness
            
            if correspondence_closeness > CONF_corr_closeness:
                E, _ = cv2.findEssentialMat(pts_img1, pts_img2, mtx)
                #pts_img1_corrected, pts_img2_corrected = cv2.correctMatches()
                whatsthis , R, T, mask = cv2.recoverPose(E, pts_img1, pts_img2, mtx)
                test_R1, test_R2, test_T = cv2.decomposeEssentialMat(E)               
                
#            axis_cam1 = np.eye(3)
#            loc_cam1 = np.zeros(shape=(3,3))
#            loc_cam_prev, axis_cam_prev = campts_hist[-1]
                loc_cam_prev, axis_cam_prev = loc_world, axis_world
                axis_cam_now = np.dot(axis_cam_prev, R)
                loc_cam_now = loc_cam_prev + np.dot(np.dot(axis_cam_prev,T), np.ones(shape=(1,3)))
                campts_hist.append((loc_cam_now, axis_cam_now))
                if len(campts_hist) > CONF_campts_hist_len:
                    campts_hist.pop(0)        
        
                # axis_cam and loc_cam represent the rotation and translation of the camera axis w.r.t. world. 
                proj_cam_prev = np.dot(mtx, \
                            np.hstack( [axis_cam_prev.T, np.dot(-axis_cam_prev.T, loc_cam_prev[:,0][np.newaxis].T)] ) )
                proj_cam_now  = np.dot(mtx, \
                            np.hstack( [axis_cam_now.T, np.dot(-axis_cam_now.T, loc_cam_now[:,0][np.newaxis].T)] ) )
                featureCloud  = cv2.triangulatePoints(proj_cam_prev, proj_cam_now, pts_img1.T, pts_img2.T)            
                featureCloud  = featureCloud/featureCloud[-1,:] # from HC to XYZ

                 # compute projection matrices
                xm,ym,zm,hm = np.median(featureCloud,axis=1)
                print 'median of cloud locations',xm,ym,zm
                if zm < 0: # when this is negative, flipping xyz to -x,-y,-z also works... i guess there isn't enough information to recover precisely
                    print mask
                    print whatsthis
                    pass


                # order by distance to camera                
                featureCloud[-1,:] = featureCloud[0,:]**2 + featureCloud[1,:]**2 + featureCloud[2,:]**2
                featureCloudSortOrder = featureCloud[-1,:].argsort()
                featureCloud = featureCloud[:, featureCloudSortOrder]
                matched_des = matched_des[featureCloudSortOrder,:]
                featureCloud = featureCloud[:-1,:] 
                
                #closestfeaturesDist = featureCloud[-1,:10]
                #closestfeaturesDes = matched_des[:10,:]                
                #closestfeaturesDist =  closestfeaturesDist/closestfeaturesDist[0]
                #print 'closest 10 features'
                #print closestfeaturesDist
                #print closestfeaturesDes
                if referencePhotogrammetricModel is None:                
                    normfactor = np.median(np.linalg.norm(featureCloud, axis=0))/10.0
                    T = T / normfactor
                    featureCloud = featureCloud / normfactor
                    referencePhotogrammetricModel = (featureCloud,matched_des)
                    normfactor = 1.0
                else:
                    normfactor = normalizePhotogrammetricModel(bf, referencePhotogrammetricModel, (featureCloud, matched_des))
                
                print 'before normalizing %f'%np.median(np.linalg.norm(featureCloud,axis=0))
                #normalize as per reference
                # do I just divide by ratio?
                T = T / normfactor
                featureCloud = featureCloud / normfactor
                print 'after normalizing %f'%np.median(np.linalg.norm(featureCloud,axis=0))                
                
                
                #sanity:
                testfactor = normalizePhotogrammetricModel(bf, referencePhotogrammetricModel, (featureCloud, matched_des))
                print 'norm factor is %f, test factor is %f'%(normfactor, testfactor)
            

               
                
                ClearAndResize(ax2, ((-50,50),(-50,50),(0,50)))
                PlotQuiver(ax2, np.vstack([loc_world, axis_world]).T, 10)
                PlotQuiver(ax2, np.vstack([loc_cam_now, axis_cam_now]).T, 10)
                PlotPtCloud(ax2, featureCloud)
                pass
    
    cv2.imshow('win1',img)
    if cv2.waitKey(66) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
plt.close('all')