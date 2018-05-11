import os
import dlib
import glob
import numpy as np
from skimage import io
import json
from PIL import ImageFile
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True,
	help="path to image folder")
args = vars(ap.parse_args())

faces_folder_path = args['folder']

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

array, descriptors, listdescriptor = [],[],[]

print()
print('Building model.json ...')
print()

for entry in os.scandir(faces_folder_path):
    if entry.is_dir():
        name = entry.name
        array.append(name)
        path = os.path.join(faces_folder_path + '/' + name, "*.jpg")
        print(name)
        
        is_empty = True
        for f in glob.glob(path):
            try:
                img = io.imread(f)            
                print('\t' + os.path.basename(f))
            
                rects = detector(img, 1)
                for k, d in enumerate(rects):
                    shape = predictor(img, d)
                face_descriptor = facerec.compute_face_descriptor(img, shape, 5)
                listdescriptor.append(list(face_descriptor))
                is_empty = False
            except:
                print('Cannot train file: {}'.format(f))
        
        if is_empty == True:
            listdescriptor = []
                
        descriptors.append(listdescriptor)
        listdescriptor = []
 
model = [{"className": c, "faceDescriptors": f} for c, f in zip(array, descriptors)]

with open(faces_folder_path + '/model.json', 'w') as outfile:
    json.dump(model, outfile)

print()
print('Build model.json completed.')
