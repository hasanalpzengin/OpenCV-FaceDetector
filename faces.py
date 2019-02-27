import numpy as np
import cv2
import pickle
from mtcnn.mtcnn import MTCNN

face_detector = MTCNN()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {}
with open("labels.pickle", "rb") as f:
	org_labels = pickle.load(f)
	labels = {v:k for k,v in org_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
	# Capture frame by frame
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_detector.detect_faces(frame)

	# recognize with keras tensorflow pytorch scikit learn

	for face in faces:
		#prepare
		roi_gray = gray[face['box'][0]: face['box'][0]+face['box'][2], face['box'][1]: face['box'][1]+face['box'][3]]
		#resize face
		cv2.resize(roi_gray, (550, 550))
		#recognize
		id_, conf = recognizer.predict(roi_gray)
		if conf>=40:
			cv2.putText(
				frame,
				labels[id_],
				(face['box'][0],face['box'][1]),
				cv2.FONT_HERSHEY_COMPLEX,
				1,
				(255,255,255),
				1
			)

		#draw rectangle
		cv2.rectangle(frame, (face['box'][0],face['box'][1]), (face['box'][0]+face['box'][2],face['box'][1]+face['box'][3]), (0,255,0), 2)
	# Display the results frame
	cv2.imshow('Face Window', frame)
	if(cv2.waitKey(20) & 0xFF == ord('q')):
		break

cap.release()
cv2.destroyAllWindows()