import cv2
from roboflow import Roboflow
import numpy as np
roboflow = Roboflow(api_key="0FXBnpOwj18EhiRWQI6z")
model = roboflow.model('project.version(1)')()

def preprocess(image):
    image = cv2.resize(image, (300, 300))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image, dtype=np.float32)
    image /= 255.0
    image = np.expand_dims(image, axis=0)
    return image
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # preprocess the image
    input_image = preprocess(frame)
    
    # make the prediction
    prediction = model.predict(input_image)
    
    # get the class and score
    class_id = prediction['detection_classes'][0]
    score = prediction['detection_scores'][0]
    
    # check if the class is Nike or Fake Nike
    if class_id == 0:
        label = "Fake Nike"
    elif class_id == 1:
        label = "Nike"
    else:
        label = "Unknown"
    
    # draw the label and score on the image
    cv2.putText(frame, f"{label}: {score:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # show the image
    cv2.imshow("Image", frame)
    
    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
