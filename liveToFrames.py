import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# Opens the Video file
cap = cv2.VideoCapture(0)
i = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    if ret == False:
        break
    for (x, y, w, h) in faces:
        print(x, y, w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+h]
        cv2.imwrite('mihir' + str(i) + '.jpg', frame)
        print('raxit' + str(i) + '.jpg saved')
        cropped = roi_color
        cv2.imwrite('cropped' + str(i) + '.jpg', cropped)
        i += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
