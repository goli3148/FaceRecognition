import os
from FaceRecognition import FaceRecognition
import cv2 as cv
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time
from modAL.models import ActiveLearner
import matplotlib.pyplot as plt
import pickle
from ImageAugmentation import ImageAugmentor


class Model:
    def __init__(self, labels_num=20, labels_each_min=10, labels_each_max=30):
        self.training_data_params = [labels_num, labels_each_min, labels_each_max]
        self.IMAGES_LABLED_DATA  = '/media/mrj/documents/AI/FaceRecognition/data/labaled_data/images'
        self.IMAGES_UNLABLED_DATA  = '/media/mrj/documents/AI/FaceRecognition/data/unlabled_data/images'
        self.Face = FaceRecognition()
        self.LABELS = None
        self.X = []
        self.Y = []
        self.learner: ActiveLearner = None
        self.face_embedding_model = InceptionResnetV1(pretrained='vggface2').eval()
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.image_augmentor = ImageAugmentor()

    def train(self):
        self.training_data_selection()
        self.embedding_extraction()
        self.model_training()
        self.save_model()
        self.load_model()

    def evaluation(self, image=None, image_path=None):
        if image_path != None:
            image = cv.imread(image_path)

        image_rgb = image.copy()
        image = cv.cvtColor(image_rgb, cv.COLOR_BGR2GRAY)
        faces, boxes = self.Face.detect_haar(image, None)
        poses, aligned_faces = self.Face.pose_and_align(image_rgb, boxes)
        X = []
        if aligned_faces is not None:
            for aligned_face in aligned_faces:
                
                image_rgb = cv.cvtColor(aligned_face, cv.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
                image_tensor = self.transform(image_pil).unsqueeze(0)
                with torch.no_grad():
                    embdeding = self.face_embedding_model(image_tensor).squeeze().numpy()
                X.append(embdeding)
        confidences = []
        predicts = []
        for x in X:
            probas = self.learner.predict_proba(x.reshape(1, -1))
            predict = self.learner.predict(x.reshape(1, -1))
            confidences.append(max(probas[0]))
            predicts.append(predict)
        return confidences, predicts

    def ActiveLearning(self, image=None, image_path=None, Threshold=0.2):
        if image_path != None:
            image = cv.imread(image_path)
        
        confidences, predicts = self.evaluation(image=image)
        confidence = confidences[0]
        predict = predicts[0]
        if confidence < Threshold:
            new_label = len(np.unique(self.Y)) + 1
            images = self.image_augmentor(image=image, unlabeled_path=self.IMAGES_UNLABLED_DATA, new_label=new_label, samples=100)
        elif confidence < 0.4:
            new_label = predict
            images = self.image_augmentor(image=image, unlabeled_path=self.IMAGES_UNLABLED_DATA, new_label=new_label, samples=10)

        X = []
        Y = []
        for image in images:
            image_rgb = image.copy()
            image = cv.cvtColor(image_rgb, cv.COLOR_BGR2GRAY)
            faces, boxes = self.Face.detect_haar(image, None)
            poses, aligned_faces = self.Face.pose_and_align(image_rgb, boxes)
            if aligned_faces is not None:
                for aligned_face in aligned_faces:
                    image_rgb = cv.cvtColor(aligned_face, cv.COLOR_BGR2RGB)
                    image_pil = Image.fromarray(image_rgb)
                    image_tensor = self.transform(image_pil).unsqueeze(0)
                    with torch.no_grad():
                        embdeding = self.face_embedding_model(image_tensor).squeeze().numpy()
                    X.append(embdeding)
                    Y.append(new_label)
        
        self.learner.teach(X, Y)
            

    def training_data_selection(self):
        label_num, label_each_min, label_each_max = self.training_data_params
        LABELS = os.listdir(self.IMAGES_LABLED_DATA)
        NUM_DATA_EACH_LABEL = {}
        for index, label in enumerate(LABELS):
            NUM_DATA_EACH_LABEL[label] = len(os.listdir(os.path.join(self.IMAGES_LABLED_DATA, label)))
        sorted_dict = dict(sorted(NUM_DATA_EACH_LABEL.items(), key=lambda item: item[1], reverse=False))

        LABELS = []
        for key, value in sorted_dict.items():
            if  label_each_max > value > label_each_min:
                LABELS.append(key)
            if len(LABELS) == label_num:
                break
        
        self.LABELS = LABELS
        return
    
    def embedding_extraction(self):
        for i, lable in enumerate(self.LABELS):
            print(f'\r{len(self.LABELS)}:{i}', end='')
            images_lable_path = os.path.join(self.IMAGES_LABLED_DATA, lable)
            
            for image_name in os.listdir(images_lable_path):
                image_path = os.path.join(images_lable_path, image_name)

                image_rgb = cv.imread(image_path)
                image = cv.cvtColor(image_rgb, cv.COLOR_BGR2GRAY)

                faces, boxes = self.Face.detect_haar(image, None)
                poses, aligned_faces = self.Face.pose_and_align(image_rgb, boxes)
                if aligned_faces is not None:
                    for aligned_face in aligned_faces:
                        
                        image_rgb = cv.cvtColor(aligned_face, cv.COLOR_BGR2RGB)
                        image_pil = Image.fromarray(image_rgb)
                        image_tensor = self.transform(image_pil).unsqueeze(0)
                        with torch.no_grad():
                            embdeding = self.face_embedding_model(image_tensor).squeeze().numpy()
                        self.X.append(embdeding)
                        self.Y.append(lable)
        
    def model_training(self):
        classifier = RandomForestClassifier()
        self.learner = ActiveLearner(classifier, X_training=self.X, y_training=self.Y)

    def save_model(self):
        pickle.dump(self.learner, open('../models/classic_online.pkl', 'wb'))

    def load_model(self):
        self.learner = pickle.load(open('../models/classic_online.pkl', 'rb'))
