import pathlib
import dlib
import os,sys
import pickle
import json
import cv2

dirpath = os.path.realpath(os.path.dirname(sys.argv[0]))

train_folder = os.path.join(dirpath, "train", "thangdc.pkl")

unknown_name = "Unknown"
threshold_recognition = 5

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(dirpath, 'models/shape_predictor_5_face_landmarks.dat'))
facerec = dlib.face_recognition_model_v1(os.path.join(dirpath, 'models/dlib_face_recognition_resnet_model_v1.dat'))

faceProto = os.path.join(dirpath, 'models/opencv_face_detector.pbtxt')
faceModel = os.path.join(dirpath, 'models/opencv_face_detector_uint8.pb')
ageProto = os.path.join(dirpath, 'models/age_deploy.prototxt')
ageModel = os.path.join(dirpath, 'models/age_net.caffemodel')
genderProto = os.path.join(dirpath, 'models/gender_deploy.prototxt')
genderModel = os.path.join(dirpath, 'models/gender_net.caffemodel')

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
padding = 40

if os.path.isfile(train_folder):
    print('Loading {} model'.format(train_folder))
    with open(train_folder, 'rb') as f:
        (le, clf) = pickle.load(f)
