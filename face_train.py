import numpy as np
import cv2
import time
from mtcnn.mtcnn import MTCNN
import math
import os
import pickle
import shutil

from PIL import Image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

BASE_DIR = os.path.dirname(os.path.abspath('__file__'))

ROOT_IMAGE_DIR = os.path.join(BASE_DIR, 'images')
if not os.path.exists(ROOT_IMAGE_DIR):
	os.mkdir(ROOT_IMAGE_DIR)

face_name = input("Please enter the name of face (press enter to pass scan): ")
print('Progress Started..')

face_detector = MTCNN();

if len(face_name)>1: 
	IMAGE_DIR = os.path.join(ROOT_IMAGE_DIR, face_name)
	if not os.path.exists(IMAGE_DIR):
		os.mkdir(IMAGE_DIR)
	else:
		shutil.rmtree(IMAGE_DIR)
		os.mkdir(IMAGE_DIR)

	cap = cv2.VideoCapture(0)

	image = np.zeros((512, 726, 3), np.uint8)
	image.fill(255)

	cv2.putText(
		image,
		"You have 10 seconds to train dataset",
		(0,256),
		cv2.FONT_HERSHEY_COMPLEX_SMALL,
		1,
		(0,0,0),
		1
	)

	cv2.putText(
		image,
		"move your face while we are scanning it.",
		(0,300),
		cv2.FONT_HERSHEY_COMPLEX_SMALL,
		1,
		(0,0,0),
		1
	)


	cv2.cvtColor(image, cv2.COLOR_BGR2BGR555)
	cv2.imshow('Introduction', image)

	cv2.waitKey(5000)

	cv2.destroyWindow('Introduction')
	t_end = time.time() + 10;

	while(time.time() < t_end):
		# Capture frame by frame
		ret, frame = cap.read()
		faces = face_detector.detect_faces(frame)
		
		cv2.putText(
			frame,
			str(math.ceil(t_end-time.time())),
			(100,100),
			cv2.FONT_HERSHEY_COMPLEX,
			1,
			(0,255,0),
			1
		)
		# is just one face to train
		if not len(faces)>1:
			cv2.imwrite(os.path.join(IMAGE_DIR, "face_"+str(time.time())+".jpg"), frame);
			cv2.rectangle(frame, (faces[0]['box'][0], faces[0]['box'][1]), (faces[0]['box'][0]+faces[0]['box'][2], faces[0]['box'][1]+faces[0]['box'][3]), (0,255,0), 2)
		else:
			cv2.putText(
				frame,
				"More than one face detected",
				(100,100),
				cv2.FONT_HERSHEY_COMPLEX,
				1,
				(0,255,0),
				1
			)

		cv2.imshow('Face Window', frame)

		if(cv2.waitKey(20) & 0xFF == ord('q')):
			break

	cap.release()
	cv2.destroyAllWindows()

#train faces
recognizer = cv2.face.LBPHFaceRecognizer_create()
y_labels = []
x_train = []
current_id = 0
label_ids = {}

for root, dirs, files in os.walk(ROOT_IMAGE_DIR):
	for file in files:
		if file.endswith('png') or file.endswith('jpeg') or file.endswith('jpg'):
			path = os.path.join(root, file)
			label = os.path.basename(root).replace(' ', "-").lower()

			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			
			id_ = label_ids[label]
			#numpy array
			org_image = cv2.imread(path)
			gray_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2GRAY)
			faces = face_detector.detect_faces(org_image)

			cv2.rectangle(org_image, (faces[0]['box'][0], faces[0]['box'][1]), (faces[0]['box'][0]+faces[0]['box'][2], faces[0]['box'][1]+faces[0]['box'][3]), (0,255,0), 2)
			cv2.imshow("Face", org_image)

			if(cv2.waitKey(50) & 0xFF == ord('q')):
				break

			for face in faces:
				roi = gray_image[face['box'][0]: face['box'][0]+face['box'][2], face['box'][1]: face['box'][1]+face['box'][3]]
				#resize face
				resized_roi = cv2.resize(roi, (550, 550))
				x_train.append(resized_roi)
				y_labels.append(id_)

with open("labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)
	
recognizer.train(x_train, np.array(y_labels))
recognizer.save('trainner.yml')

print("Progress Finished...")