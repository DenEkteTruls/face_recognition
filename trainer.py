import cv2
import numpy as np
import os

def train():
    path = 'dataset'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("frontalface.xml")

    def getImageAndLabel(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []

        for imagePath in imagePaths:
            image = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_RGB2GRAY)
            image_numpy = np.array(image, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(image_numpy)
            for(x, y, w, h) in faces:
                faceSamples.append(image_numpy[y:y+h, x:x+w])
                ids.append(id)
        return faceSamples, ids

    print("\n [ INFO ] Training faces. It will take a few seconds. Wait ...")

    faces, ids = getImageAndLabel(path)
    recognizer.train(faces, np.array(ids))
    recognizer.write('data.yml')

    print(f"\n [ INFO ] {len(np.unique(ids))} faces trained. Exiting the trainer.")