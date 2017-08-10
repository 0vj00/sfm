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
from solveForH import *

def queryCamProperties(cap):
    print 'width %d'%cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print 'height %d'%cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
def ClearAndResize(ax, lims):
     ax.cla()
     ax.set_xlim3d(*lims[0])
     ax.set_ylim3d(*lims[1])
     ax.set_zlim3d(*lims[2])

def drawRvecsTvecs(ax, rvecs, tvecs):
     ClearAndResize(ax,((-1,1),(-1,1),(-1,1)))
     #ax.quiver(*(np.vstack([np.zeros(shape=(3,1)), rvecs])), pivot = 'tail', color = 'red')
     ax.quiver(*(np.vstack([np.zeros(shape=(3,1)), tvecs])), pivot = 'tail', color = 'blue')        
     plt.draw()
     
def PlotQuiver(ax, mat, l):
    assert(mat.shape == (3,6))
    c = ['red', 'green', 'black']
    for i in range(3):
        ax.quiver3D(*mat[i], pivot = 'tail', length = l, color = c[i])
    plt.draw()

def PlotPtCloud(ax, ptcld):
    #print 'ptcld shape',ptcld.shape
    assert(ptcld.shape[0] == 3)
    ax.scatter(ptcld[0,:],ptcld[1,:],ptcld[2,:])
    plt.draw()

def trackFeatures(img1, img2, kp_track):
    pts1_ft_idx, pts1 = kp_track.get_curr_kp()
    pts2, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, pts1, None, \
                                winSize = (21,21), \
                                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001))
    status = status.reshape(-1)
    pts2 = pts2[(status > 0),:]
    kp_track.add_tracked_kp(pts1_ft_idx[(status>0)], pts2)
    pts1 = pts1[(status > 0),:]
    return pts1, pts2
    
def computeResidual(E, pts1, pts2, mtxinv):
    F = np.dot(mtxinv.T, np.dot(E, mtxinv))
    numpts = pts1.shape[0]
    pts1HC = np.hstack([pts1, np.ones(shape=(numpts,1))])
    pts2HC = np.hstack([pts2, np.ones(shape=(numpts,1))])
    return np.mean(np.dot( pts1HC, np.dot(F, pts2HC.T)))
        
class FeatureTrack:
    def __init__(self, numf, numt):
        self.numf = numf
        self.numt = numt
        self.untracked_kp = np.ones(shape=(self.numf,))
        self.ft_mat = -1*np.ones(shape=(self.numf, self.numt, 2),dtype='float32')
        self.curr_t = 0        
        self.ft_stats = np.zeros(self.ft_mat.shape[0])
        self.ft_stats2 = np.zeros(self.ft_mat.shape[1]+1)
        
    def add_new_kp(self, kp_list, next_slot = False):
        num_new_kp = kp_list.shape[0]
        if num_new_kp > 0:
            if next_slot:
                self.inc_t()
            num_avail_kp = np.sum(self.untracked_kp)
            assert( num_avail_kp >= num_new_kp )
            kp_idx = np.where(self.untracked_kp == 1)[0]
            kp_idx = kp_idx[:num_new_kp]
            self.ft_mat[kp_idx, :, :] = -1 # since this is a new feature, clear it's history
            self.ft_mat[kp_idx, self.curr_t, :] = kp_list
            self.untracked_kp[kp_idx] = 0
    
    def add_tracked_kp(self, kp_idx, kp_list, next_slot = True):
        assert(np.sum(self.untracked_kp[kp_idx])==0)
        if next_slot:        
            self.inc_t()
        self.ft_mat[kp_idx, self.curr_t, :] = kp_list
        self.untracked_kp[:] = 1
        self.untracked_kp[kp_idx] = 0
        self.ft_mat[np.where(self.untracked_kp == 1)[0], :, :] = -1        
        
    def get_curr_kp(self):
        curr_tracked_f = np.where(self.untracked_kp == 0)[0]        
        return (curr_tracked_f, self.ft_mat[curr_tracked_f, self.curr_t, :].reshape(-1,2))
        
    def inc_t(self):
        self.curr_t = (self.curr_t + 1)%self.numt
    
    def ret_dec_t(self, ip):
        return ( (ip -1 + self.numt)%self.numt )
    
    def ret_inc_t(self, ip):
        return ( (ip +1)%self.numt )
    
    def stats(self):
        self.ft_stats = np.zeros(self.ft_mat.shape[0])
        self.ft_stats2 = np.zeros(self.ft_mat.shape[1]+1)
        for i in xrange(self.ft_mat.shape[0]):
            self.ft_stats[i] = np.sum( self.ft_mat[i,:,0] > -1 )
            self.ft_stats2[self.ft_stats[i]] += 1
            #if ft_stats[i] > 0:
            #    print 'feature %d has %d entries'%(i,ft_stats[i])
        #print 'num features with ... entries'
        #print self.ft_stats2
    
    def get_kp_across_imgs(self):
        self.stats()
        valid_f, valid_f_pts_curr = self.get_curr_kp()
        min_hist_ = int(np.min(self.ft_stats[valid_f]))
        t = self.curr_t
        for i in range(min_hist_ - 1):
            t = self.ret_dec_t(t)
        valid_f_pts_past = self.ft_mat[valid_f, t, :].reshape(-1,2)
        return (valid_f, valid_f_pts_curr, valid_f_pts_past, min_hist_)
    
    def draw_tracks(self, img):
        self.stats()
        d = 100/self.numt
        colorlist = [(0, 255-100+d*i, 255-d*i) for i in range(self.numt)]
        valid_f, _ = self.get_curr_kp()
        for fcntr in valid_f:
            hist_ = int( self.ft_stats[fcntr] )
            if hist_ > 1:
                t = self.curr_t
                for i in range(hist_ - 1):
                    t = self.ret_dec_t(t)
                newpts = self.ft_mat[fcntr, t, :]
                for i in range(hist_ - 1):
                    refpts = newpts            
                    t = self.ret_inc_t(t)
                    newpts = self.ft_mat[fcntr, t, :]
                    cv2.line(img, tuple(refpts), tuple(newpts),color=colorlist[i] )
        return


def extractFeatures(ffd, img):
    def ffd_extract(ffd, imgcrop):
        return ffd.detect(imgcrop, None)
    def goodfeaturesextract(imgcrop):
        kp_imgcrop = cv2.goodFeaturesToTrack(imgcrop,5,0.1,2)
        if kp_imgcrop is None:
            return None
        kp_imgcrop = kp_imgcrop.reshape(-1,2)
        kp_imgcrop = [cv2.KeyPoint(x,y,2) for x,y in kp_imgcrop]
        return kp_imgcrop
        
    h, w = img.shape[:2]
    kplist = []
    hinc = 100
    winc = 100
    for i in np.arange(0,h,hinc):
        for j in np.arange(0,w,winc):
            deltah = np.min([hinc, h-i])
            deltaw = np.min([winc, w-j])            
            imgcrop = img[i:i+deltah, j:j+deltaw]
            kp_imgcrop = ffd_extract(ffd, imgcrop)
            #kp_imgcrop = goodfeaturesextract(imgcrop)
            if kp_imgcrop is None:
                continue
            for k in kp_imgcrop:
                k.pt = (k.pt[0]+j, k.pt[1]+i)
            kp_imgcrop = sorted(kp_imgcrop, key=lambda x: -x.response)
            kplist.extend(kp_imgcrop[:])
    #kplist = ffd.detect(img, None)
    return kplist
                
if 'cap' in globals():
    cap.release()
cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
assert( cap.isOpened() )

plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
fig2 = plt.figure()
ax2 = fig2.add_subplot(111,projection='3d')


mtx_b4_undistort, dist = cPickle.load(open('microsoftHD3Kcalib.mtxdist','rb'))
mtx_b4_undistort_inv = np.linalg.inv( mtx_b4_undistort )
mtx, roi=cv2.getOptimalNewCameraMatrix(mtx_b4_undistort,dist,(640,480),0,(640,480))
mtxinv = np.linalg.inv(mtx)

ffd = cv2.FastFeatureDetector_create(nonmaxSuppression = True, threshold = 30)

coord_world = [np.zeros(shape=(3,3)), np.eye(3)]
prevImg = None
prevFeaturePts = None
matchedprevFeaturePts = None

#leftcamframe = cPickle.load(open('leftcam.frm','rb'))
#rightcamframe = cPickle.load(open('rightcam.frm','rb'))
img_hist = [];
for i in xrange(5):
    ret,img = cap.read()
    assert(ret)
    
    img = cv2.undistort(img, mtx_b4_undistort, dist, None, mtx)
    x,y,w,h = roi
    img = img[y:y+h, x:x+w]

    img_hist.append(img)
lastGoodT2 = np.zeros((3,))
#tmp_T_accum = np.zeros(shape=(3,1))
stable_kp = None
kp_track = FeatureTrack(2000,50)
cntr1 = 0
while True:
    cntr1 = (cntr1+1)%5
    img = img_hist[-1].copy()     
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret2, img2 = cap.read()
    img2 = cv2.undistort(img2, mtx_b4_undistort, dist, None, mtx)
    x,y,w,h = roi
    img2 = img2[y:y+h, x:x+w]    
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    img_hist.append(img2.copy())
    img_hist.pop(0)
    
    if stable_kp is None:
        kp = extractFeatures(ffd, imggray)        
        cam1FeaturePts = np.asarray([k.pt for k in kp]).astype('float32')
        kp_track.add_new_kp(cam1FeaturePts)
    
    matchedcam1FeaturePts, matchedcam2FeaturePts = trackFeatures(imggray, img2gray, kp_track)

    stable_kp = matchedcam2FeaturePts    
    
    matchedkp  = [cv2.KeyPoint(x,y,2) for x,y in matchedcam1FeaturePts]
    matchedkp2 = [cv2.KeyPoint(x,y,2) for x,y in matchedcam2FeaturePts]
    
    matchedFeatureSetSize = matchedcam2FeaturePts.shape[0]
    matchedFeaturePtsXdelta, matchedFeaturePtsYdelta = np.mean(matchedcam2FeaturePts - matchedcam1FeaturePts,axis=0)
    matchedFeaturePtsdelta = np.mean(np.linalg.norm(matchedcam2FeaturePts - matchedcam1FeaturePts,axis=1))
    kp_track.stats()
    
    #img  = cv2.drawKeypoints(img,  matchedkp,  None, color=(0,0,255))
    img2 = cv2.drawKeypoints(img2, matchedkp2, None, color=(0,0,255))  

    if matchedFeatureSetSize > 5 and matchedFeaturePtsdelta > 2:
            (valid_f, pts_curr_img, pts_prev_img, howfarback) = kp_track.get_kp_across_imgs()
            print howfarback
            HList = solveForH(pts_prev_img, pts_curr_img, mtxinv)
            numH = len(HList)
            colorlist = [(255-255+15*i,255-15*i, 0) for i in range(10)]
            cntr = 0
            for _,l in HList:
                lkp = [cv2.KeyPoint(x,y,2) for x,y in pts_curr_img[l]]
                img = cv2.drawKeypoints(img, lkp, None, color = colorlist[cntr])                
                cntr += 1
                if cntr == 10:
                    break
            img = cv2.putText(img, '# H: '+str(numH), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255)  )            
            
            if numH > 5:
                E2, _ = cv2.findEssentialMat(pts_curr_img, pts_prev_img, mtx, method = cv2.RANSAC, threshold=1)
                _, R2, T2, mask2 = cv2.recoverPose(E2, pts_curr_img, pts_prev_img, mtx)
                img = cv2.putText(img, '% : '+str(np.round(100*np.sum(mask2 > 0)/mask2.shape[0])), (50,150), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255)  )                
                if np.sum(mask2 > 0)/mask2.shape[0] > 0.95:
                    lastGoodT2 = T2
                
                    T2 = 10*T2
                    loc_cam_prev, axis_cam_prev = coord_world[0], coord_world[1]
                    axis_cam_now = np.dot(axis_cam_prev, R2)
                    loc_cam_now = loc_cam_prev + np.dot(np.dot(axis_cam_prev,T2), np.ones(shape=(1,3)))
                    proj_cam_prev = np.dot(mtx, \
                            np.hstack( [axis_cam_prev.T, np.dot(-axis_cam_prev.T, loc_cam_prev[:,0][np.newaxis].T)] ) )
                    proj_cam_now  = np.dot(mtx, \
                            np.hstack( [axis_cam_now.T, np.dot(-axis_cam_now.T, loc_cam_now[:,0][np.newaxis].T)] ) )
                    featureCloud  = cv2.triangulatePoints(proj_cam_prev, proj_cam_now, pts_prev_img.T, pts_curr_img.T)            
                    featureCloud  = featureCloud/featureCloud[-1,:] # from HC to XYZurr
                    featureCloud = featureCloud[:3,:]
               
                    ClearAndResize(ax2, ((-50,50),(-50,50),(0,50)))
                    PlotQuiver(ax2, np.vstack([loc_cam_prev, axis_cam_prev]).T, 10)
                    PlotQuiver(ax2, np.vstack([loc_cam_now, axis_cam_now]).T, 10)
                    PlotPtCloud(ax2, featureCloud)
                img = cv2.putText(img, 'T2 : '+str(np.round(lastGoodT2,1)), (50,100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255)  ) 
                
            pass
    kp_track.draw_tracks(img2)
    img3 = np.hstack([img, img2])
    cv2.imshow('win1', img3)
    if cv2.waitKey(66) & 0xFF == ord('q'):
        break
    # to do : add new pts to here...
    if stable_kp.shape[0] < 300:
        kp = extractFeatures(ffd, img2gray)
        if len(kp) > 0:
            detectedFeaturePts = np.asarray([k.pt for k in kp]).astype('float32')
            stable_kp_r = np.round(stable_kp)
            f_in_kp_1d = np.sum(np.in1d(detectedFeaturePts,stable_kp_r).reshape(-1,2), axis=1)
            newf = np.where(f_in_kp_1d < 2)
            additionalFeaturePts = detectedFeaturePts[newf]
            #kp_track.stats()
            kp_track.add_new_kp(additionalFeaturePts)
            #kp_track.stats()
            _, stable_kp = kp_track.get_curr_kp()
        
        
cap.release()
cv2.destroyAllWindows()
