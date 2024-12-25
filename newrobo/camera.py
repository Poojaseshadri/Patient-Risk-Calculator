import cv2
import numpy as np
from roboflow import Roboflow

rf = Roboflow(api_key="0FXBnpOwj18EhiRWQI6z")
project = rf.workspace().project("fake-logo-detection-rss3q")
model = project.version(1).model

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # convert frame to buffer
    _, buffer = cv2.imencode('.jpg', frame)
    
    # predict on buffer
    response = model.predict(buffer, confidence=40, overlap=30)

    for pred in response['predictions']:
        x, y, w, h = pred['bbox']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
