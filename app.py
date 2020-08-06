import cv2
import numpy as np
import os, sys
import time
import trainer

try: os.remove("dataset/.DS_Store")
except: pass

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('data.yml')
faceCascade = cv2.CascadeClassifier("frontalface.xml")
font = cv2.FONT_HERSHEY_SIMPLEX
path = 'dataset'

def confidenceToPercent(confidence):
    return confidence+20

def getNewUserId():
    ps = [os.path.join(path, f) for f in os.listdir(path)]
    nums = []
    for p in ps:
        nums.append(int(os.path.split(p)[-1].split(".")[1]))
    if nums:
        return np.max(nums)+1
    else:
        return 1

id = 0

names = []
with open("names.txt", "r") as f:
    data = (f.read()).split(",")
    names += data

print(names)

cam = cv2.VideoCapture(int(sys.argv[1]))
cam.set(3, 640)
cam.set(4, 480)

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

lastOfLast = [0, 0, 0, 0, "", 0]
last = [0, 0, 0, 0, "", 0]

start = time.time()

def register(image):
    username = str(input("Username: "))
    snaps = []

    print("\n [ INFO ]Scanning your face ...")

    for i in range(10):
        time.sleep(1)
        newImage = cam.read()[1]
        cv2.imshow("frame", newImage)
        snaps.append(newImage)

    with open("names.txt", "a+") as f:
        f.write(", "+username)

    newUserId = getNewUserId()

    for pos, snap in enumerate(snaps):
        cv2.imwrite(f"{path}/User."+str(newUserId)+"."+str(pos+1)+".jpg", snap)

    print("Done\n\n [ INFO ] Starting the trainer from the app.")

    trainer.train()

while True:

    image = cv2.flip(cam.read()[1], 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH))
    )

    end = time.time()
    if end-start < 3:
        lastOfLast = last
        cv2.rectangle(image, (last[0], last[1]), (last[0]+last[2], last[1]+last[3]), (0, 255, 100), 1)
        cv2.putText(image, f"{str(last[4])} [ {str(confidenceToPercent(last[5]))}% ]", (last[0]-20, last[1]-10), font, 1, (0, 255, 255), 2)


    for (x, y, w, h) in faces:
        start = time.time()
        #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 100), 1)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if(confidence < 100):
            try:
                id = names[id]
            except:
                id = "unknown"
            confidence = round(100 - confidence)
        else:
            id = "unknown"
            confidence = round(100 - confidence)

        last = [x, y, w, h, id, confidence]

        #cv2.putText(image, f"{str(id)} [ {str(confidenceToPercent(confidence))}% ]", (x+5, y-5), font, 1, (255, 255, 255), 1

    cv2.imshow("frame", image)

    if cv2.waitKey(10) & 0xFF == ord("r"):
        register(image)

print("\n [ INFO ] Exiting the program and cleaning up stuff")
cam.release()
cv2.destroyAllWindows()