import os
from common import utils, config, data
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GMM
from sklearn.model_selection import GridSearchCV
import cv2
import argparse
import pickle

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=False, default="train",
	help="path to image folder")
ap.add_argument("-c", "--classifier", required=False,  default="LinearSvm",
	help="Training method: LinearSvm, GridSearchSvm, RadialSvm, DecisionTree, GaussianNB, DBN")
args = vars(ap.parse_args())

faces_folder_path = args['folder']
classifier = args['classifier']

labels = []
descriptors = []

start = time.time()
    
for file in data.iterImgs(faces_folder_path):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    try:
        img = file.getRGB()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        face_locations = config.detector(gray, 1)
        print("\tFound {} face(s) from \"{}\" file".format(len(face_locations), file.path))

        face_descriptors = [np.asarray(list(config.facerec.compute_face_descriptor(img, shape))) for shape in [config.predictor(gray, d) for d in face_locations]]
        
        if len(face_locations) == 1:
            for descriptor in face_descriptors:
                descriptors.append(list(descriptor))
            labels.append(file.cls)
        else:
            if file.name.lower() != "unknown":
                print("\tThis {} file has no or more than one images, so it will be deleted.".format(file.path))
                os.remove(file.path)
                print('\tRemoving {} file'.format(file.path))
    except Exception as error:
        os.remove(file.path)
        print('\tRemoving {} because {}'.format(file.path, error))
        pass

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

fName = "{}/model_{}.pkl".format(config.train_folder, classifier)
print("Saving classifier to '{}'".format(fName))
with open(fName, 'wb') as f:
    pickle.dump((le, clf), f)
