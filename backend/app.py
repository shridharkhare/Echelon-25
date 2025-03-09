import os
from flask import Flask, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Load YOLO model
model = YOLO("model_v5.pt")  # Replace with your model path

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return {"error": "No file uploaded"}, 400

    file = request.files['file']
    
    # Read image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert OpenCV BGR to RGB

    # Run inference
    results = model.predict(image, conf=0.1, iou=0.5)

    # Annotate the image
    annotated_image = results[0].plot()  # Draw bounding boxes and labels

    # Ensure 'static' directory exists
    static_folder = "static"
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)

    # Save the image
    output_path = os.path.join(static_folder, "annotated_result.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    # Return image URL
    return {"image_url": "http://127.0.0.1:8000/static/annotated_result.jpg"}

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
