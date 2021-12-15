import pathlib
import dlib
import os,sys
import pickle
import json
from os.path import exists as file_exists

dirpath = os.path.realpath(os.path.dirname(sys.argv[0]))

#train_folder = os.path.join(dirpath, "train")
train_folder = os.path.join(dirpath, "train", "model.pkl")

unknown_name = "Unknown"
threshold_recognition = 5

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(dirpath, 'models/shape_predictor_5_face_landmarks.dat'))
facerec = dlib.face_recognition_model_v1(os.path.join(dirpath, 'models/dlib_face_recognition_resnet_model_v1.dat'))

print('Loading {} model'.format(train_folder))
if file_exists(train_folder):
    with open(train_folder, 'rb') as f:
        (le, clf) = pickle.load(f)
