import os
from FaceDetection import Face
import cv2 as cv
from facenet_pytorch import InceptionResnetV1
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

START_TIME = time.time()

ALIGNED_LABLED_DATA = '/media/mrj/documents/AI/FaceRecognition/data/labaled_data/aligned'
EMBEDD_LABLED_DATA = '/media/mrj/documents/AI/FaceRecognition/data/labaled_data/embedding'
IMAGES_LABLED_DATA  = '/media/mrj/documents/AI/FaceRecognition/data/labaled_data/images'

ALIGNED_UNLABLED_DATA = '/media/mrj/documents/AI/FaceRecognition/data/unlabled_data/aligned'
EMEBDD_UNLABLED_DATA = '/media/mrj/documents/AI/FaceRecognition/data/unlabled_data/embedding'
IMAGES_UNLABLED_DATA  = '/media/mrj/documents/AI/FaceRecognition/data/unlabled_data/images'


# FACE DETECTION AND ALIGNMENT 
LABLES = os.listdir(IMAGES_LABLED_DATA)

Face = Face()

for i, lable in enumerate(LABLES):
    images_lable_path = os.path.join(IMAGES_LABLED_DATA, lable)
    aligned_lable_path = os.path.join(ALIGNED_LABLED_DATA, lable)
    if not os.path.exists(aligned_lable_path):
        os.mkdir(aligned_lable_path)

    for image_name in os.listdir(images_lable_path):
        image_path = os.path.join(images_lable_path, image_name)

        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        faces, boxes = Face.detect_haar(image, None)
        poses, aligned_faces = Face.pose_and_align(image, boxes)
        if aligned_faces is not None:
            for aligned_face in aligned_faces:
                cv.imwrite(os.path.join(aligned_lable_path, image_name), aligned_face)
    if i == 100:
        break

FACE_TIME = time.time()
print(f"Face time:{FACE_TIME - START_TIME}")

# FACE EMBEDDING EXTRACTION
model = InceptionResnetV1(pretrained='vggface2').eval()
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


LABLES = os.listdir(ALIGNED_LABLED_DATA)

for i, lable in enumerate(LABLES):
    aligned_lable_path = os.path.join(ALIGNED_LABLED_DATA, lable)
    embedd_lable_path = os.path.join(EMBEDD_LABLED_DATA, lable)
    if not os.path.exists(embedd_lable_path):
        os.mkdir(embedd_lable_path)

    for image_name in os.listdir(aligned_lable_path):
        image_path = os.path.join(aligned_lable_path, image_name)

        image = cv.imread(image_path)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image_pil = Image.fromarray(image_rgb)
        image_tensor = transform(image_pil).unsqueeze(0)

        with torch.no_grad():
            embdeding = model(image_tensor).squeeze().numpy()
        
        np.save(os.path.join(embedd_lable_path, image_name),embdeding)

EMBEDD_TIME = time.time()
print(f"Embedding time:{EMBEDD_TIME - FACE_TIME}")

# CLASSIFICATION MODEL
LABLES = os.listdir(EMBEDD_LABLED_DATA)
X, Y = [], []
for i, lable in enumerate(LABLES):
    embedd_lable_path = os.path.join(EMBEDD_LABLED_DATA, lable)
    for embedd in os.listdir(embedd_lable_path):
        Y.append(i)
        X.append(np.load(os.path.join(embedd_lable_path, embedd)))

svc = SVC()
svc.fit(X, Y)
print(accuracy_score(Y, svc.predict(X)))

CLASSIFICATION_TIME = time.time()
print(f"Classificaion time: {CLASSIFICATION_TIME-EMBEDD_TIME}")
print(f"whole time:{CLASSIFICATION_TIME-START_TIME}")
