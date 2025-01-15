import cv2 as cv
import numpy as np
import dlib
import openface
from facenet_pytorch import InceptionResnetV1
import torch
from PIL import Image
from torchvision import transforms
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner
import pickle

class FaceRecognition:
    def __init__(self):
        self.haar_cascade = cv.CascadeClassifier('/media/mrj/documents/AI/FaceRecognition/models/haarcascade_frontalface_default.xml')
        self.face_pose_predictor = dlib.shape_predictor("/media/mrj/documents/AI/FaceRecognition/models/shape_predictor_68_face_landmarks.dat")
        self.face_aligner = openface.AlignDlib("/media/mrj/documents/AI/FaceRecognition/models/shape_predictor_68_face_landmarks.dat")
        self.vgg_face_model = InceptionResnetV1(pretrained='vggface2').eval()
        try:
            self.load_classifier()
        except:
            print("No classifier found")

        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def load_classifier(self):
        self.active_learner:ActiveLearner = pickle.load(open('/media/mrj/documents/AI/FaceRecognition/models/learner.pkl', 'rb'))

    def save_classifier(self):
        pickle.dump(self.active_learner, open('/media/mrj/documents/AI/FaceRecognition/models/learner.pkl', 'wb'))

    def detect_haar(self, reference_img, save_dir=None):
        faces_geo = self.haar_cascade.detectMultiScale(reference_img, 1.1, 9)
        faces = []
        boxes = []
        for index, (x, y, w, h) in enumerate(faces_geo): 
            faces.append(reference_img[y:y+h, x:x+w])
            if save_dir is not None:
                cv.imwrite(f"{save_dir}/face{index}.jpg", reference_img[y:y+h, x:x+w]) 
            boxes.append(dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h))
        return faces, boxes
    
    def detect_dlib(self, reference_img, save_dir=None):
        face_detector = dlib.get_frontal_face_detector()
        detected_faces = face_detector(reference_img, 1)
        boxes = []
        faces = []
        for index, face_rect in enumerate(detected_faces):
            x0, y0, x1, y1 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()
            faces.append(reference_img[y0:y1, x0:x1])
            boxes.append(face_rect)
            if save_dir is not None:
                cv.imwrite(f"{save_dir}/face{index}.jpg", reference_img[y0:y1, x0:x1])
        
        return faces, boxes
    
    def pose_and_align(self, frame, boxes):
        poses = []
        alignedFaces = []
        for i in range(len(boxes)):
            pose = self.face_pose_predictor(frame, boxes[i])
            points = [[p.x, p.y] for p in pose.parts()]
            poses.append(points)
            alignedFace = self.face_aligner.align(534, frame, boxes[i], landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            alignedFaces.append(alignedFace)
        return np.array(poses), alignedFaces
    
    def extract_embeddings(self, aligned_images):
        embeddings = []
        for aligned_image in aligned_images:
            if len(aligned_image.shape)==2:
                aligned_image = cv.cvtColor(aligned_image, cv.COLOR_GRAY2RGB)
            pil_image = Image.fromarray(aligned_image)
            tensor_image = self.transform(pil_image).unsqueeze(0)
            with torch.no_grad():
                embedding = self.vgg_face_model(tensor_image).squeeze().numpy()
            embeddings.append(embedding)
        return np.array(embeddings)
    
    def predict(self, X):
        return self.active_learner.predict(X)
    
    def confidence(self, X):
        confidences = []
        for x in X:
            x = x.reshape(1, -1)
            probas = self.active_learner.predict_proba(x)
            confidence = max(probas[0])
            confidences.append(confidence)
        return np.array(confidences)
    
    def teach(self, X, Y):
        self.active_learner.teach(X=X, y=Y)