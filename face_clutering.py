import dlib
import cv2
import argparse
from common import utils, config, data

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True,
	help="path to image folder")
ap.add_argument("-e", "--extract", required=True, 
	help="path to extract folder")
ap.add_argument("-t", "--threshold", required=False, default=0.4,
	help="threshold")
args = vars(ap.parse_args())

folder = args['folder']
extracted = args['extract']
threshold = float(args['threshold'])

descriptors = []
images = []

for f in data.iterImgs(folder):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print('')
    print("Processing file: {}".format(f.path))
    try:
        img = f.getRGB()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = config.detector(gray, 1)
          
        for k, d in enumerate(dets):
            if d.width() > 100 and d.height() > 100:
                shape = config.predictor(img, d)
                face_descriptor = config.facerec.compute_face_descriptor(img, shape)
                descriptors.append(face_descriptor)              
                images.append((img, d))
    except Exception as error:
        print(error)
        pass
labels = dlib.chinese_whispers_clustering(descriptors, threshold)
for i, label in enumerate(labels):
    img, d = images[i]
    name = "User_" + str(label)
    face_img = img[d.top() : d.top() + d.height(), d.left() : d.left() + d.width()]        
    utils.save_face_image(face_img, name, extracted, 0)
