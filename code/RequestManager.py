import requests
import cv2 as cv
import datetime
import os

class RequestManager:
    def __init__(self):
        self.url = 'http://localhost:5000//api'

    def sendNewImage(self, image, label):
        file_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        file_dir = os.path.join('code', f'{file_name}.jpg')
        cv.imwrite(file_dir, image)
        file = {'file' : open(file_dir, 'rb')}
        data = {'label': label}
        response = requests.post(url=f'{self.url}/UnlabelFace', data=data, files=file)
        if response.status_code == 200:
            ...
    
    def getNewLabel(self):
        response = requests.get(url=f'{self.url}/NewLabel')
        return response.json()['new label']

