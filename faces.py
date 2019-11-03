import numpy as np
import cv2
import pickle
from mtcnn.mtcnn import MTCNN

#get face detector again
face_detector = MTCNN()

#get recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
# read trained model (trained by face_train.py)
recognizer.read("trainner.yml")

labels = {}
# get labels
'''
	that is works to conversation like:
	1 -> Hasan Alp Zengin
	2 -> Dan Reynolds
	because our trained model will give output as integer not string
'''
with open("labels.pickle", "rb") as f:
	org_labels = pickle.load(f)
	labels = {v:k for k,v in org_labels.items()}

# get webcam
cap = cv2.VideoCapture(0)

# while program running
while(True):
	# get webcam frame
	ret, frame = cap.read()
	# gray scale again to detect and recognize face
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces in frame
	faces = face_detector.detect_faces(frame)

	# for each face
	for face in faces:
		# get only face (area of interest)
		roi_gray = gray[face['box'][0]: face['box'][0]+face['box'][2], face['box'][1]: face['box'][1]+face['box'][3]]
		#resize face
		cv2.resize(roi_gray, (550, 550))
		# Here we are calling recognizer to predict detected face
		id_, conf = recognizer.predict(roi_gray)
		# Conf is the similarity rate in this example I hold it in minimum 40
		# it can be changed if we need high success rate
		if conf>=40:
			# write face's name
			cv2.putText(
				frame,
				labels[id_],
				(face['box'][0],face['box'][1]),
				cv2.FONT_HERSHEY_COMPLEX,
				1,
				(255,255,255),
				1
			)

		#draw rectangle around the face
		cv2.rectangle(frame, (face['box'][0],face['box'][1]), (face['box'][0]+face['box'][2],face['box'][1]+face['box'][3]), (0,255,0), 2)
	# Display the results frame
	cv2.imshow('Face Window', frame)
	if(cv2.waitKey(20) & 0xFF == ord('q')):
		break

cap.release()
cv2.destroyAllWindows()