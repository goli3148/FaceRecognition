import requests

url = 'http://localhost:5000//api'

def post():
    url = 'http://localhost:5000//api/UnlabelFace'

    csrf_toke = 'you-will-never-guess'
    files = {'file': open('/media/mrj/documents/AI/FaceRecognition/data/labaled_data/images/Aaron_Eckhart/Aaron_Eckhart_0001.jpg', 'rb')}
    data = {'label': 5}
    response = requests.post(url, files=files, data=data)
    print(response.text)

def newlabel():
    response = requests.get(url=f'{url}/NewLabel')
    return response.json()['new label']

print(newlabel())