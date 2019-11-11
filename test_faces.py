import sys, os
import cv2
import numpy as np
import pickle
from mtcnn.mtcnn import MTCNN

#get face detector again
face_detector = MTCNN()
#define result variables
success = 0;
fail = 0;

#get recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
# read trained model (trained by face_train.py)
recognizer.read("trainner.yml")

def getFolderName(path):
    return path.split("/")[1]

labels = {}
# get labels
with open("labels.pickle", "rb") as f:
	org_labels = pickle.load(f)
	labels = {v:k for k,v in org_labels.items()}

path = "images"
for path,subdirs,files in os.walk(path):
    for file in files:
        #expected result
        expected = getFolderName(path)
        filePath = os.path.join(path, file)
        image = cv2.imread(filePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detect_faces(image)

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
            if(conf>=40):
                result = labels[id_]
                if(result!=expected):
                    fail += 1
                else:
                    success +=1
            else:
                fail += 1

print("Result: Fail=", fail, " Success=", success)