import cv2
import numpy as np

cap = cv2.VideoCapture(1)

assert( cap.isOpened() )

while True:
    ret, frame = cap.read()
    cv2.imshow('win1',frame)
    if( cv2.waitKey(33) & 0xFF == ord('q') ):
        break
    
cap.release()
cv2.destroyAllWindows()