from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os, shutil
from ultralytics import YOLO
import mimetypes
mimetypes.init()

app = Flask(__name__)

uploadFolder = os.path.join('static', 'images')

# app.config['UPLOAD'] = uploadFolder

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # save the uploaded file
        file = request.files['uploadedMedia']
        filename = secure_filename(file.filename)
        uploadedMedia = os.path.join(uploadFolder, filename)
        file.save(uploadedMedia)

        # call yolov8 model
        model = YOLO("best.pt")

        # check uploaded file type with mimetype to see if the file is image or video
        fileType = mimetypes.guess_type(uploadedMedia)[0]

        print(fileType)

        if fileType != None:
            fileType = fileType.split('/')[0]

        # check if predict path exist
        if os.path.exists(os.path.join(uploadFolder, "predict")):
            shutil.rmtree(os.path.join(uploadFolder, "predict")) # if predict path exxist then remove it
        
        # predict and save result in predict path
        model.predict(uploadedMedia, save=True, project=uploadFolder, name="predict")
        annotated_media = os.path.join(uploadFolder, "predict", filename)

        return render_template('index.html', result=annotated_media, type=fileType)
        
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)