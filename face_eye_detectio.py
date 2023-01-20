import cv2
import numpy as np

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')

vdo= cv2.VideoCapture(0)
while(True):
    flag, img = vdo.read()
    if(flag==False):
        break
    cv2.putText(img,"press 'x' to close..",(10,30),cv2.FONT_HERSHEY_PLAIN,2,(216,60,147),2)
    
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceModel.detectMultiScale(gray_img,minSize=(100,100))
    
    for x,y,w,h in faces:
        cv2.rectangle(img, (x,y,w,h),(219,50,147),1)
        
        face = gray_img[y:y+h,x:x+w]
        eyes = eyeModel.detectMultiScale(face)
        
        for ex,ey,ew,eh in eyes:
            cv2.rectangle(img, (x+ex,y+ey,ew,eh), (221,160,221),2)
        
              
    cv2.imshow("Face & Eye Detection",img)
    key = cv2.waitKey(7)
    if (key== ord('x')):
        break
cv2.destroyAllWindows()
vdo.release()
