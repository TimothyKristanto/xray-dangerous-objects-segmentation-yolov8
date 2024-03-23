from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os, shutil
from ultralytics import YOLO

app = Flask(__name__)

upload_folder = os.path.join('static', 'images')

app.config['UPLOAD'] = upload_folder


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img = os.path.join(app.config['UPLOAD'], filename)
        model = YOLO("best.pt")
        if os.path.exists(os.path.join(app.config['UPLOAD'], "predict")):
            shutil.rmtree(os.path.join(app.config['UPLOAD'], "predict"))
        model.predict(img, save=True, project=app.config['UPLOAD'], name="predict")
        annotated_img = os.path.join(app.config['UPLOAD'], "predict", filename)
        return render_template('index.html', img=annotated_img)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)