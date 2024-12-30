import cv2 as cv
import numpy as np
from FaceDetection import Face
import os

FD = Face()
camera = cv.VideoCapture(0)

if not camera.isOpened:
    print("Error: Could not open cmera")
    exit()

print("Press q to quit")

frame_ignores = 40
counter = 0
while True:
    ret, frame = camera.read()

    if not ret:
        print("Error: Failed to capture frame")
        break

    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 

    faces, boxes = FD.detect_haar(gray_img, None)
    poses, aligned_faces = FD.pose_and_align(gray_img, boxes)

    if poses is not None:
        for pose in poses[0]:
            gray_img = cv.circle(frame, (pose[0], pose[1]), radius=1, thickness=1, color=(0,0,255))
        
        for aligned_face in aligned_faces:
            gray_img = aligned_face

    cv.imshow("",gray_img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break