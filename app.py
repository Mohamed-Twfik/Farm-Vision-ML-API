from flask import Flask, jsonify, request, send_from_directory, g
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
from tensorflow import keras
import numpy as np
import cv2 

import os

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://farm_vision:Z1Y18QfiqtO92YxVTM0nfl4m3eKZS3d4@dpg-cgoqqsd269v5rjd53ul0-a.frankfurt-postgres.render.com/farm_vision"
db = SQLAlchemy(app)

imagesFolderURL = "files/modelsImages/"

def removeFile(url):
    try:
        if os.path.exists(url):
            os.remove(url)
            print("File deleted successfully")
        else:
            print("File not found")
    
    except Exception as e:
        print("Remove File error: " + str(e))


def diseaseDetectionModel(imagename):
    try:
        # model processes
        img = Image.open(imagesFolderURL+imagename)
        img = img.resize((640, 640)) # resize to match model input size
        img.save(imagename)
        model = YOLO('models/best.pt')
        results = model.predict(stream=True, imgsz=640, source=imagename ,save=True, project=imagesFolderURL, name="", exist_ok=True)
        output= []
        for r in results:
            for c in r.boxes.cls:
                output.append(model.names[int(c)])
        # unique output values
        output = set(output)
        output = list(output)
        removeFile(imagename)
        # Move the image from runs folder to images folder
        resultImageName = "result-"+imagename
        os.rename(imagesFolderURL+"predict/"+imagename, imagesFolderURL+resultImageName)
        os.rmdir(imagesFolderURL+"predict/")
        return output, resultImageName
    
    except Exception as e:
        return "Disease Detection Model error: " + str(e), 500

def plantClassificationModel(imagename):
    try:
        # preprocessing part
        img = cv2.imread(imagesFolderURL+imagename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(imagename, gray)
        test = []
        img = cv2.imread(imagename)
        resized_img = cv2.resize(img, (256,256))
        test.append(resized_img)
        test = np.array(test)
        test_scaled = test / 255

        # model processes part
        class_namesAll = ['Maize', 'Strawberry', 'Wheat']
        modelAll = keras.models.load_model('models/3PlantFinal.h5')
        y_pred = modelAll.predict(test_scaled)
        class_name = class_namesAll[np.argmax(y_pred)]
        removeFile(imagename)
        return class_name
    
    except Exception as e:
        return "Plant Classification Model error: " + str(e), 500


@app.before_request
def authentication():
    try:
        auth_token = request.headers.get('x-auth-token')
        if auth_token is None:
            return "Token is required", 401
        getTokenQuery = text('SELECT "Tokens"."token", "Tokens"."UserId" FROM public."Tokens" WHERE "Tokens"."token"=:token')
        result = db.session.execute(getTokenQuery, {"token": auth_token})
        tokenData = result.mappings().all()
        if not tokenData:
            return "Token is invalid", 401
        g.tokenData = tokenData[0]

    except Exception as e:
        return "authentication error: " + str(e), 500


@app.before_request
def imageValidation():
    try:
        if request.path.startswith('/api/imagesModel'):
            if "image" not in request.files:
                return "No file part", 400
            image = request.files["image"]
            if image.filename == "":
                return "No selected file", 400
            
            # save the file with a secure filename
            now = datetime.now()
            timestamp = now.timestamp()
            milliseconds = round(timestamp * 1000)
            imagename = "image-" + str(milliseconds) + ".jpg"
            image.save(imagesFolderURL+imagename)
            g.imagename = imagename

    except Exception as e:
        return "image validation error: " + str(e), 500


@app.route('/')
def hello_world():
    return 'Hello, World!'
