import os
import cv2
import numpy as np
from PIL import Image
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, "images")

current_id = 0
labels = {}
userids_for_labels = []
images_for_labels = []

image_size = (1024, 720)

print("Model training started!")
print("Please wait...")
for root, directories, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png"):
            path = os.path.join(root, file)

            # Get user name for a corresponding picture, and capitalize the name
            label = os.path.basename(os.path.dirname(path)).lower().capitalize()

            # Construct the labels object with their corresponding ids
            if not label in labels:
                labels[label] = current_id
                current_id += 1

            user_id = labels[label]

            # Convert image to grayscale and resize before adding them to the array
            grayscale_image = Image.open(path).convert("L")
            resized_image = grayscale_image.resize(image_size, Image.ANTIALIAS)
            image_array = np.array(resized_image, "uint8")

            faces = face_cascade.detectMultiScale(image_array)
            for(x, y, w, h) in faces:
                region_of_interest = image_array[y:y+h, x:x+w]
                images_for_labels.append(region_of_interest)
                userids_for_labels.append(user_id)

with open("labels.pickle", 'wb') as f:
    pickle.dump(labels, f)

recognizer.train(images_for_labels, np.array(userids_for_labels))
recognizer.save("trained-faces.yml")

print("Training finished successfully!")
