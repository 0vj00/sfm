import cv2
import numpy as np

cap0 = cv2.VideoCapture(0)
#cap1 = cv2.VideoCapture(1)

assert( cap0.isOpened() )
#assert( cap1.isOpened() )

while True:
    ret0, frame0 = cap0.read()
    cv2.imshow('win1',frame0)
 #   ret1, frame1 = cap1.read()
  #  cv2.imshow('win2',frame1)

    if( cv2.waitKey(33) & 0xFF == ord('q') ):
        break
    
cap0.release()
#cap1.release()
cv2.destroyAllWindows()