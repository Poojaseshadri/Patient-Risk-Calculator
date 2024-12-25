from flask import Flask, render_template, request, url_for
from roboflow import Roboflow
import os
import numpy as np
from PIL import Image

app = Flask(__name__)

rf = Roboflow(api_key="0FXBnpOwj18EhiRWQI6z")
project = rf.workspace().project("fake-logo-detection-rss3q")
model = project.version(1).model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image = request.files['file'].read()
        image_np = np.frombuffer(image, np.uint8)
        max_size = 65500
        if max(image_np.shape[:2]) > max_size:
            ratio = float(max_size) / max(image_np.shape[:2])
            new_size = tuple([int(x*ratio) for x in image_np.shape[:2]])
            image_np = np.array(Image.fromarray(image_np).resize(new_size, Image.BICUBIC))
        response = model.predict(image_np, confidence=40, overlap=30).json()
        model.predict(image_np, confidence=40, overlap=30).json()
        predictions = response['predictions']
        if predictions and len(predictions) > 0:
            predicted_class = predictions[0]['class']
            if predicted_class == 'nike' or predicted_class == 'fakenike':
                predicted_image = 'predicted.jpg'
                with open(os.path.join(app.static_folder, predicted_image), 'wb') as f:
                    f.write(model.predict(image, confidence=40, overlap=30).content)
                return render_template('index.html', prediction=predicted_class, predicted_image=predicted_image)
            else:
                return render_template('index.html', prediction="Not a Nike or a fake Nike")
        else:
            return render_template('index.html', prediction="No prediction found")
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
