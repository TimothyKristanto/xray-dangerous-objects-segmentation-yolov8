from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os, shutil
from ultralytics import YOLO
import mimetypes
from natsort import natsorted
import cv2

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

        if fileType != None:
            fileType = fileType.split('/')[0]

        # check if predict path exist
        if os.path.exists(os.path.join(uploadFolder, "predict")):
            shutil.rmtree(os.path.join(uploadFolder, "predict")) # if predict path exist then remove it
        
        # predict and save result in predict path
        if fileType == "image":
            model.predict(uploadedMedia, save=True, project=uploadFolder, name="predict")
        elif fileType == "video":
            model.predict(uploadedMedia, save=True, save_frames=True, project=uploadFolder, name="predict")
            createVideoFromImages(uploadFolder + "/predict/" + filename.split('.')[0] + "_frames/", filename, uploadFolder + "/predict")

        annotated_media = "predict/" + filename

        return render_template('index.html', result=annotated_media, type=fileType)
        
    return render_template('index.html')

def createVideoFromImages(source, output, outFolder):
    img = []

    for i in os.listdir(source):
        i = source + "/" + i
        img.append(i)

    img = natsorted(img)

    cv2_fourcc = cv2.VideoWriter_fourcc(*'avc1')

    frame = cv2.imread(img[0])
    size = list(frame.shape)
    del size[2]
    size.reverse()

    video = cv2.VideoWriter(outFolder + "/" + output, cv2_fourcc, 30, size)

    for i in range(len(img)):
        video.write(cv2.imread(img[i]))

    video.release()

if __name__ == '__main__':
    app.run(debug=True)