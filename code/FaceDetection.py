import cv2 as cv
import numpy as np
import dlib
import openface

class Face:
    def __init__(self):
        self.haar_cascade = cv.CascadeClassifier('code/haarcascade_frontalface_default.xml')
        self.face_pose_predictor = dlib.shape_predictor("code/shape_predictor_68_face_landmarks.dat")
        self.face_aligner = openface.AlignDlib("code/shape_predictor_68_face_landmarks.dat")


        
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
        if len(boxes)==0:
            return None, None
        poses = []
        alignedFaces = []
        for i in range(len(boxes)):
            pose = self.face_pose_predictor(frame, boxes[i])
            points = [[p.x, p.y] for p in pose.parts()]
            poses.append(points)
            alignedFace = self.face_aligner.align(534, frame, boxes[i], landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            alignedFaces.append(alignedFace)
        return np.array(poses), alignedFaces
            
