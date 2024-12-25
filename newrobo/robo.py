from roboflow import Roboflow
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
rf = Roboflow(api_key="0FXBnpOwj18EhiRWQI6z")
project = rf.workspace().project("fake-logo-detection-rss3q")
model = project.version(1).model

# infer on a local image
#print(model.predict("38.jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("41.jpg", confidence=40, overlap=30).save("prediction.jpg")
img = mpimg.imread('prediction.jpg')
plt.imshow(img)
plt.show()
# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())