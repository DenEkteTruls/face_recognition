import os

ids = []
images = []

for a in os.listdir("dataset"):
    ids.append(a)
    image_set = []
    for b in os.listdir(os.path.join("dataset", a)):
        image_set.append(b)
    images.append(image_set)
        
print(ids)
print(images)
