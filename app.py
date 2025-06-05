from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from torchvision import transforms
import cv2
import numpy as np

app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar modelo (ejemplo con YOLOv5)
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

model = load_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Procesar imagen
            results = process_image(filepath)
            
            return render_template('results.html', 
                                filename=filename,
                                objects=results['objects'],
                                counts=results['counts'],
                                image_with_boxes=results['image_with_boxes'])
    
    return render_template('index.html')

def process_image(image_path):
    img = Image.open(image_path)
    results = model(img)
    detections = results.pandas().xyxy[0]
    
    # Contar objetos
    object_counts = {}
    for obj in detections['name']:
        object_counts[obj] = object_counts.get(obj, 0) + 1
    
    # Dibujar bounding boxes
    img_with_boxes = results.render()[0]
    img_with_boxes = Image.fromarray(img_with_boxes)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated_' + os.path.basename(image_path))
    img_with_boxes.save(output_path)
    
    return {
        'objects': detections['name'].unique().tolist(),
        'counts': object_counts,
        'image_with_boxes': 'annotated_' + os.path.basename(image_path)
    }

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)