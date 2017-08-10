
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

    return np.median(ratios)

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        pt1 = pt1.astype(np.int)
        pt2 = pt2.astype(np.int)
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

plt.close('all')
fig = plt.figure()
fig2 = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax2 = fig2.add_subplot(111,projection='3d')

# Configuration
CONF_keypts_hist_len = 1
CONF_campts_hist_len = 1
CONF_corr_closeness = 10
CONF_samples_photogram_ratio = 10000

if 'cap' in globals():
    cap.release()
cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
assert( cap.isOpened() )

mtx, dist = cPickle.load(open('microsoftHD3Kcalib.mtxdist','rb'))
leftimg = cPickle.load(open('leftview.pkl','rb'))
rightimg = cPickle.load(open('rightview.pkl','rb'))


orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING) #crossCheck=True
i = 0

keypts_hist=[] # stores (keypoints, description, image)
campts_hist=[] # stores (location, axis, rotation_matrix), all are w.r.t world coordinates
referencePhotogrammetricModel = None

# need world to be origin and unit coords...not sure code will work if this is changed
axis_world = np.eye(3)
loc_world  = np.zeros(shape=(3,3))

campts_hist.append((loc_world, axis_world))

imgarr=[leftimg, rightimg]
#for i in [0,1]:
while True:
    i = (i+1)%10    #i = (i+1)%10
    ret, img = cap.read()ubuntu v4l2
    
    #img = imgarr[i]
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    kp, des = orb.detectAndCompute(imggray,None)    
    #print 'found %d key points'%(len(kp))
    img = cv2.drawKeypoints(img, kp, img, color=(0,255,0), flags = 0)
    
    if des is not None and len(keypts_hist) > 0:
        # try to match keypoints with those from past
        kp_prev, des_prev, img_prev = keypts_hist[0]
        assert(des_prev.dtype == des.dtype and des_prev.shape[1] == des.shape[1])
        
        #matches = bf.match(des_prev, des)
        matches = bf.knnMatch(des_prev, des, k=2)
        
        matches = [m[0] for m in matches if len(m) > 0 and m[0].distance < 0.7*m[1].distance]
        #print 'found %d matches; %f, %f percent'%(len(matches), 100*len(matches)/keypts_hist[0][1].shape[0], 100*len(matches)/des.shape[0])
        matches = [m for m in matches if m.distance < 20]        
        matched_kp = [kp[m.trainIdx] for m in matches]
        matched_kp_in_hist = [kp_prev[m.queryIdx] for m in matches]
        matched_des = np.asarray([des[m.trainIdx] for m in matches])
        
        img_hist = img_prev
        img_hist = cv2.drawKeypoints(img_hist, keypts_hist[0][0], img_hist, color=(0,255,0), flags=0)
#        for i in range(len(matches)):
#            img = cv2.drawKeypoints(img, [matched_kp[i]], img, color=(0,0,255), flags = 0)
#            img_hist = cv2.drawKeypoints(img_hist, [matched_kp_in_hist[i]], img_hist, color=(0,0,255), flags=0)
#            cv2.imshow('tmp',np.hstack([img,img_hist]))
#            if cv2.waitKey(0) & 0xFF == ord('q'):
#                break
                
        
        
        if len(matched_kp) > 5:
        
            pts_img1 = np.asarray([k.pt for k in matched_kp_in_hist])
            pts_img2 = np.asarray([k.pt for k in matched_kp])
            correspondence_closeness = np.abs(pts_img1-pts_img2).sum() / pts_img1.shape[0]
            print 'corrspondence pts closeness %d pix per pt'%correspondence_closeness
            
            if correspondence_closeness > CONF_corr_closeness:
                E, _ = cv2.findEssentialMat(pts_img1, pts_img2, mtx)
                #F, mask = cv2.findFundamentalMat(pts_img1, pts_img2, cv2.FM_8POINT)
                mtxinv = np.linalg.inv(mtx)                
                F = np.dot(np.dot(mtxinv.T,E),mtxinv)

                #mask = mask.reshape(-1)
                #pts_img1 = pts_img1[mask > 0, :]
                #pts_img2 = pts_img2[mask > 0, :]
                #matched_des = matched_des[mask>0,:]
                img_histgray=cv2.cvtColor(img_hist, cv2.COLOR_BGR2GRAY)
                lines1 = cv2.computeCorrespondEpilines(pts_img2.reshape(-1,1,2), 2, F)
                lines1 = lines1.reshape(-1,3)
                img_hist, _ = drawlines(img_histgray,imggray,lines1,pts_img1,pts_img2)                
                lines2 = cv2.computeCorrespondEpilines(pts_img1.reshape(-1,1,2), 1, F)
                lines2 = lines2.reshape(-1,3)
                img, _ = drawlines(imggray, img_histgray, lines2, pts_img2, pts_img1)
#                
                #pts_img1_corrected, pts_img2_corrected = cv2.correctMatches()
                _ , R, T, mask = cv2.recoverPose(E, pts_img1, pts_img2, mtx)
                test_R1, test_R2, test_T = cv2.decomposeEssentialMat(E)               
                
                loc_cam_prev, axis_cam_prev = loc_world, axis_world
                axis_cam_now = np.dot(axis_cam_prev, R)
                loc_cam_now = loc_cam_prev + np.dot(np.dot(axis_cam_prev,T), np.ones(shape=(1,3)))
        
                # axis_cam and loc_cam represent the rotation and translation of the camera axis w.r.t. world. 
                proj_cam_prev = np.dot(mtx, \
                            np.hstack( [axis_cam_prev.T, np.dot(-axis_cam_prev.T, loc_cam_prev[:,0][np.newaxis].T)] ) )
                proj_cam_now  = np.dot(mtx, \
                            np.hstack( [axis_cam_now.T, np.dot(-axis_cam_now.T, loc_cam_now[:,0][np.newaxis].T)] ) )
                featureCloud  = cv2.triangulatePoints(proj_cam_prev, proj_cam_now, pts_img1.T, pts_img2.T)            
                featureCloud  = featureCloud/featureCloud[-1,:] # from HC to XYZ

                xm,ym,zm,hm = np.median(featureCloud,axis=1)
                print 'median of cloud locations',xm,ym,zm
                if zm < 0: # when this is negative, flipping xyz to -x,-y,-z also works... i guess there isn't enough information to recover precisely
                    #print mask
                    pass
                
                # order by distance to camera                
                featureCloud = featureCloud[:-1,:] 
                
                if referencePhotogrammetricModel is None:                
                    normfactor = np.median(np.linalg.norm(featureCloud, axis=0))/10.0
                    T = T / normfactor
                    featureCloud = featureCloud / normfactor
                    referencePhotogrammetricModel = (featureCloud,matched_des)
                    normfactor = 1.0
                else:
                    normfactor = normalizePhotogrammetricModel(bf, referencePhotogrammetricModel, (featureCloud, matched_des))
                
                print 'before normalizing %f'%np.median(np.linalg.norm(featureCloud,axis=0))
                T = T / normfactor
                featureCloud = featureCloud / normfactor
                print 'after normalizing %f'%np.median(np.linalg.norm(featureCloud,axis=0))                
                #sanity:
                testfactor = normalizePhotogrammetricModel(bf, referencePhotogrammetricModel, (featureCloud, matched_des))
                print 'norm factor is %f, test factor is %f'%(normfactor, testfactor)
            
                axis_cam_now = np.dot(axis_cam_prev, R)
                loc_cam_now = loc_cam_prev + np.dot(np.dot(axis_cam_prev,T), np.ones(shape=(1,3)))
                campts_hist.append((loc_cam_now, axis_cam_now))
                if len(campts_hist) > CONF_campts_hist_len:
                    campts_hist.pop(0)        
               
                ClearAndResize(ax2, ((-50,50),(-50,50),(0,50)))
                PlotQuiver(ax2, np.vstack([loc_world, axis_world]).T, 10)
                PlotQuiver(ax2, np.vstack([loc_cam_now, axis_cam_now]).T, 10)
                PlotPtCloud(ax2, featureCloud)
                pass
            
        cv2.imshow('winhist',img_hist)

    if des is not None and i == 0:
        # store keypoints and image
        keypts_hist.append((kp,des,img))
        if len(keypts_hist) > CONF_keypts_hist_len:
            keypts_hist.pop(0)    
    
    cv2.imshow('win1',img)
    if cv2.waitKey(66) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
plt.close('all')
