import cv2
import os

cap = cv2.VideoCapture(0)

if not os.path.exists('dataset/vietdai'):
    os.makedirs('dataset/vietdai')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face = frame[y:y + h, x:x + w]
        cv2.imwrite(f'dataset/vietdai/person_{x}.jpg', face)

    cv2.imshow('Face', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()