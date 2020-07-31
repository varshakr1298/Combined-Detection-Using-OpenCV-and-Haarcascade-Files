import cv2
import numpy as np
smile_cascade=cv2.CascadeClassifier('haarcascades1/haarcascade_smile.xml')
face_cascade=cv2.CascadeClassifier('haarcascades1/haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascades1/haarcascade_eye.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
import cv2

cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()                        # Reading the Frame
	faces = face_cascade.detectMultiScale(frame,1.3,5)
	for (x,y,w,h) in faces:
		cv2.putText(frame,'Face',(x,y), font, 1,(0,0,255),2)
		cv2.rectangle(frame,(x,y),((x + w),(y + h)),(255,0,0),3) 
		eyes=eye_cascade.detectMultiScale(frame,1.1,5)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(frame,(ex,ey),((ex + ew),(ey + eh)),(0,255,0),3)
		smiles=smile_cascade.detectMultiScale(frame, 1.4,60)
		for (sx,sy,sw,sh) in smiles:
			cv2.rectangle(frame,(sx,sy),((sx + sw),(sy + sh)),(255,0,255),3)               
	cv2.imshow('result', frame)
	if cv2.waitKey(33) == 27:
		break;
cap.release()
cv2.destroyAllWindows()
