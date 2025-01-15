import cv2 as cv
from FaceRecognition import FaceRecognition
import numpy as np
from RequestManager import RequestManager
# ignore warnings
import warnings
warnings.filterwarnings("ignore")

class CameraFeed:
    def __init__(self):
        self.FD = FaceRecognition()
        self.camera = cv.VideoCapture(0)
        if not self.camera.isOpened:
            print("Error: Could not open camera")
            exit()
        print("Press q to quit")
        self.rm = RequestManager()
        self.FRAME_RATE_IGNORANCE = 30

    def feed(self):
        counter = 0
        while True:
            if counter == self.FRAME_RATE_IGNORANCE:
                counter = 0
            else:
                counter +=1
                continue

            ret, frame = self.camera.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
            faces, boxes = self.FD.detect_dlib(gray_img, None)
            if len(faces) == 0:
                print("\rNo Faces", end='')
                cv.imshow("",gray_img)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    self.camera.release()
                    break
                continue

            poses, aligned_faces = self.FD.pose_and_align(gray_img, boxes)
            embeddings = self.FD.extract_embeddings(aligned_faces)
            
            confidence = self.FD.confidence(embeddings)[0]
            if confidence < 0.1:
                print("\rUnknown", end='')
                new_label = self.rm.getNewLabel()
                self.FD.active_learner.teach(embeddings, np.array([200+new_label]))
                self.rm.sendNewImage(image=frame, label=new_label)
            elif confidence < 0.6:
                label = self.FD.predict(embeddings)[0]
                self.FD.active_learner.teach(embeddings, np.array([label])) 
                print(f'\r{self.FD.predict(embeddings)}', end='')
            else:
                print(f'\r{self.FD.predict(embeddings)}', end='')
            
            dlib_rect = boxes[0]
            x1, y1, x2, y2 = dlib_rect.left(), dlib_rect.top(), dlib_rect.right(), dlib_rect.bottom()
            frame = cv.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            cv.imshow("",frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                self.camera.release()
                break

if __name__ == "__main__":
    CF = CameraFeed()
    CF.feed()