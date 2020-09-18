import cv2
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"persons_name": 1}
with open("labels.pkl", 'rb') as f:
    labels = pickle.load(f)
    labels = {v:k for k,v in labels.items()}


#img = cv2.imread('ml.png')
cap = cv2.VideoCapture(2)

while cap.isOpened():
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)

    #print(faces)
    for (x, y, w, h) in faces:
        print(x, y, w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+h]

        #recognize...

        id_, conf = recognizer.predict(roi_gray)
        if conf < 90 and conf > 7:
            print(id_)
            print(labels[id_])
            color = (0,255,255)
            stroke = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            cv2.putText(img, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)



        img_item = "fd.png"
        cv2.imwrite(img_item, roi_gray)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()