import os
from common import utils
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GMM
from sklearn.model_selection import GridSearchCV
import dlib
from skimage import io
import cv2
import pickle
import argparse
import pathlib

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True,
	help="path to image folder")
ap.add_argument("-c", "--classifier", required=True,
	help="Training method: LinearSvm, GridSearchSvm, RadialSvm, DecisionTree, GaussianNB, DBN")
args = vars(ap.parse_args())

faces_folder_path = args['folder']
classifier = args['classifier']

dirpath = pathlib.Path(__file__).parent

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(dirpath, 'models/shape_predictor_68_face_landmarks.dat'))
facerec = dlib.face_recognition_model_v1(os.path.join(dirpath, 'models/dlib_face_recognition_resnet_model_v1.dat'))

labels = []
descriptors = []

start = time.time()
    
metadata = utils.load_metadata(faces_folder_path)
for file in metadata:

    arr = str(file).split(os.sep)    
    img = io.imread(str(file))
    frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    rects = detector(frame_gray, 1)
    print("Found {} face of {}".format(len(rects),arr[len(arr) - 1]))
    
    if len(rects) > 0:
        labels.append(arr[len(arr) - 2])
        for k, d in enumerate(rects):        
            shape = predictor(frame_gray, d)
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            descriptors.append(list(face_descriptor))

le = LabelEncoder().fit(labels)
labelsNum = le.transform(labels)
nClasses = len(le.classes_)

print("Training for {} classes.".format(nClasses))

if classifier == 'LinearSvm':
    clf = SVC(C=1, kernel='linear', probability=True)
elif classifier == 'GridSearchSvm':
    print("""
        Warning: In our experiences, using a grid search over SVM hyper-parameters only
        gives marginally better performance than a linear SVM with C=1 and
        is not worth the extra computations of performing a grid search.
        """)
    param_grid = [
        {'C': [1, 10, 100, 1000],
        'kernel': ['linear']},
        {'C': [1, 10, 100, 1000],
        'gamma': [0.001, 0.0001],
        'kernel': ['rbf']}
    ]
    clf = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5)
elif classifier == 'GMM':  # Doesn't work best
    clf = GMM(n_components=nClasses)

# ref:
# http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#example-classification-plot-classifier-comparison-py
elif classifier == 'RadialSvm':  # Radial Basis Function kernel
    # works better with C = 1 and gamma = 2
    clf = SVC(C=1, kernel='rbf', probability=True, gamma=2)
elif classifier == 'DecisionTree':  # Doesn't work best
    clf = DecisionTreeClassifier(max_depth=20)
elif classifier == 'GaussianNB':
    clf = GaussianNB()

clf.fit(descriptors, labelsNum)
print("Training time: ", time.time() - start)

fName = "{}/model_{}.pkl".format(dirpath, classifier)
print("Saving classifier to '{}'".format(fName))
with open(fName, 'wb') as f:
    pickle.dump((le, clf), f)
