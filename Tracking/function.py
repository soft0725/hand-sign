import cv2
import math

def get_dis(x1,x2,y1,y2,W):
    temp = (((x1-x2)**2) + ((y1-y2)**2)) * W
    return float(temp)

def motion(image, value):
    cv2.putText(
    image, text = value, org=(10, 30),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
    color=255, thickness=2)