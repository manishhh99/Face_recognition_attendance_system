import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join

# Ensure trainer directory exists
os.makedirs("trainer", exist_ok=True)

data_path = 'dataset/'
face_data = []
labels = []
label_map = {}

dirs = listdir(data_path)
for i, dir_name in enumerate(dirs):
    path = join(data_path, dir_name)
    label_map[i] = dir_name
    for file in listdir(path):
        img_path = join(path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        face_data.append(np.asarray(img, dtype=np.uint8))
        labels.append(i)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(face_data), np.asarray(labels))
model.save('trainer/trained_model.yml')
np.save('trainer/labels.npy', label_map)

print("✅ Model Trained Successfully")
