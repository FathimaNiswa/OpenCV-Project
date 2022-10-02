import cv2


img = cv2.VideoCapture(0)

while True:
    ret, frame = img.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    for x, y, w, h in eyes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Face", frame)
    cv2.waitKey(0)
