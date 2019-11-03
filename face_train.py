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

'''
Train FACE Process
-> This program will ask you face name and creates a directory with it.
-> After file creation process a dialog will be open that directs you to train your face with webcam
'''

# get directory to create train test file
BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
ROOT_IMAGE_DIR = os.path.join(BASE_DIR, 'images')
if not os.path.exists(ROOT_IMAGE_DIR):
	os.mkdir(ROOT_IMAGE_DIR)
# get face name to create directory
face_name = input("Please enter the name of face (press enter to pass scan): ")
print('Progress Started..')

'''
-> MTCNN is face detector, It's just stays to detect face it has no recognition option
-> It's the method based "https://arxiv.org/pdf/1604.02878.pdf" work by Zhang K., Zhang Z. and Li Z.
'''
face_detector = MTCNN()

if len(face_name)>1: 
	IMAGE_DIR = os.path.join(ROOT_IMAGE_DIR, face_name)
	#here we are checking is path exist or not
	if not os.path.exists(IMAGE_DIR):
		os.mkdir(IMAGE_DIR)
	else:
		shutil.rmtree(IMAGE_DIR)
		os.mkdir(IMAGE_DIR)

	# getting webcam hardware
	cap = cv2.VideoCapture(0)

	# Creating a dialog to direct user
	# For that a white screen creating and writing sentences to inform user
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
	# Show created white screen
	cv2.imshow('Introduction', image)

	# 5 seconds
	cv2.waitKey(5000)

	# after introduction done for 5 seconds
	cv2.destroyWindow('Introduction')

	t_end = time.time() + 10;

	#start timer for 10 seconds and get face images of user
	while(time.time() < t_end):
		
		#get frame from webcam
		ret, frame = cap.read()
		faces = face_detector.detect_faces(frame)
		
		#write second to window
		cv2.putText(
			frame,
			str(math.ceil(t_end-time.time())),
			(100,100),
			cv2.FONT_HERSHEY_COMPLEX,
			1,
			(0,255,0),
			1
		)
		
		# if face detected and not multi face
		if not len(faces)>1:
			#write face to directory /images/{facename}
			cv2.imwrite(os.path.join(IMAGE_DIR, "face_"+str(time.time())+".jpg"), frame);
			cv2.rectangle(frame, (faces[0]['box'][0], faces[0]['box'][1]), (faces[0]['box'][0]+faces[0]['box'][2], faces[0]['box'][1]+faces[0]['box'][3]), (0,255,0), 2)
		else:
			#if face couldn't detect or more than one face, warn user
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

'''
-> Now all face image collected process will pass to training
'''


# In this example recognizer chosen as LBPH
# There is also other options like Fisher and Eigen
recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer = cv2.face.EigenFaceRecognizer_create()
#recognizer = cv2.face.FisherFaceRecognizer_create()
y_labels = []
x_train = []
current_id = 0
label_ids = {}

'''
-> Retrieve all face images which ends with png, jpeg or jpg
-> Here program gets into images folder
'''
for root, dirs, files in os.walk(ROOT_IMAGE_DIR):
	for file in files:
		if file.endswith('png') or file.endswith('jpeg') or file.endswith('jpg'):
			path = os.path.join(root, file)
			# if saved face name has 2 word like Dan Reynolds turn it to dan-reynolds
			# and save it as label of images inside it.
			label = os.path.basename(root).replace(' ', "-").lower()

			# if label is not exist save it
			# Here we are using integer values therefore each string label should has a integer value
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			
			id_ = label_ids[label]

			# read images in this directory
			org_image = cv2.imread(path)
			# turn them to grayscale for detection and recognition process
			gray_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2GRAY)
			# detect face in image
			# this operation improves face recognition process because we are getting are of interest
			faces = face_detector.detect_faces(org_image)

			# get only face after detection
			cv2.rectangle(org_image, (faces[0]['box'][0], faces[0]['box'][1]), (faces[0]['box'][0]+faces[0]['box'][2], faces[0]['box'][1]+faces[0]['box'][3]), (0,255,0), 2)
			cv2.imshow("Face", org_image)

			if(cv2.waitKey(50) & 0xFF == ord('q')):
				break

			# now we have face and label of it so in this loop we are just saving them for training process
			for face in faces:
				# get area of interest (only face)
				roi = gray_image[face['box'][0]: face['box'][0]+face['box'][2], face['box'][1]: face['box'][1]+face['box'][3]]
				# resize only face to 550x550
				resized_roi = cv2.resize(roi, (550, 550))
				# input of training = grayscaled just face image
				# output of training = id of label(directory name)
				x_train.append(resized_roi)
				y_labels.append(id_)

# labels.pickle holds our labels to don't forget label names
with open("labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)
'''	
-> train face recognizer with prepared data
-> Now lets take a look what we have:
-> 	We got labels with directory names and created integer values for each
->  We got "only faces" with taking area of interest with face detection
'''
recognizer.train(x_train, np.array(y_labels))
# save training result as trainner.yml
recognizer.save('trainner.yml')

print("Progress Finished...")

# To run trained file: run faces.py