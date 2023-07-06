from flask import Flask, jsonify, request, send_from_directory, g
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
from tensorflow import keras
import numpy as np
import cv2 
from flask_cors import CORS
from tensorflow_addons.metrics import F1Score

import os
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
# app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:mohamed910@localhost/smart_farm"
db = SQLAlchemy(app)
CORS(app)

detectionModel = YOLO('models/disease.pt')
classificationModel = keras.models.load_model('models/plantType.h5', custom_objects={'FixedDropout': keras.layers.Dropout, 'Addons>F1Score': F1Score})
class_names = [
    'healthy apple',
    'healthy bell pepper',
    'healthy corn (maize)',
    'healthy grape',
    'healthy potato',
    'unhealthy apple',
    'unhealthy bell pepper',
    'unhealthy corn (maize)',
    'unhealthy grape',
    'unhealthy potato'
]


imagesFolderURL = "files/modelsImages/"
videosFolderURL = "files/modelVideos/"


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
        # model = YOLO('models/disease.pt')
        results = detectionModel.predict(stream=True, imgsz=640, source=imagename ,save=True, project=imagesFolderURL, name="", exist_ok=True)
        output= []
        for r in results:
            for c in r.boxes.cls:
                output.append(detectionModel.names[int(c)])
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
        img = Image.open(imagesFolderURL+imagename)
        img = img.resize((224,224))
        img = np.array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        y_pred = classificationModel.predict(img)
        class_name = class_names[np.argmax(y_pred[0])]
        confidence = float(np.max(y_pred[0]))
        # removeFile(imagename)
        if class_name in class_names:
            return [class_name, confidence]
        else:
            return False
    
    except Exception as e:
        return False


@app.before_request
def authentication():
    try:
        auth_token = request.form['x-auth-token']
        if auth_token is None:
            return "Token is required", 401
        getTokenQuery = text('SELECT "Tokens"."token", "Tokens"."UserId" FROM public."Tokens" WHERE "Tokens"."token"=:token')
        tokenResult = db.session.execute(getTokenQuery, {"token": auth_token})
        tokenData = tokenResult.mappings().all()
        # print(tokenData[0])
        # print(tokenData[0]['UserId'])
        if not tokenData:
            return "Token is invalid", 401
        getUserQuery = text('SELECT "Users"."premium", "Users"."haveFreeTrial" FROM public."Users" WHERE "Users"."id"=:id')
        userResult = db.session.execute(getUserQuery, {"id": tokenData[0]['UserId']})
        userData = userResult.mappings().all()
        # print(userData[0])
        if userData[0]['premium'] or userData[0]['haveFreeTrial']:
            g.tokenData = tokenData[0]
        else:
            return "Sorry you can't use our features please subscribe first", 401

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


@app.route("/api/imagesModel/diseaseDetection", methods=["POST"])
def diseaseDetectionEndPoint():
    try:
        tokenData = g.tokenData
        imagename = g.imagename

        # the model
        diseases, resultImageName = diseaseDetectionModel(imagename)

        # store data in database
        if len(diseases) < 1:
            diseases.append("Healthy Plant")
        insertImageDataQuery = text('INSERT INTO public."ModelsImages" ("image", "createdAt", "updatedAt", "UserId", "resultImage") VALUES (:image, :createdAt, :updatedAt, :UserId, :resultImage) RETURNING id;')
        current_date = datetime.now()
        imageData = db.session.execute(insertImageDataQuery, {
            "image":imagename,
            "createdAt":current_date,
            "updatedAt":current_date,
            "UserId": tokenData["UserId"],
            "resultImage": resultImageName
        })
        imageId = imageData.fetchone()[0]
        for disease in diseases:
            insertDiseasesQuery = text('INSERT INTO public."DetectDiseaseResults" ("diseaseType", "createdAt", "updatedAt", "ModelsImageId") VALUES (:diseaseType, :createdAt, :updatedAt, :ModelsImageId)')
            db.session.execute(insertDiseasesQuery, {
                "diseaseType":disease,
                "createdAt":current_date,
                "updatedAt":current_date,
                "ModelsImageId": imageId
            })
        db.session.commit()

        # send response
        response = {"diseases":diseases, "image":imagename, "resultImage":resultImageName}
        return jsonify(response), 200
    
    except Exception as e:
        return "Disease Detection error: " + str(e), 500


@app.route("/api/imagesModel/plantClassification", methods=["POST"])
def plantClassificationEndPoint():
    try:
        tokenData = g.tokenData
        imagename = g.imagename

        # the model
        result = plantClassificationModel(imagename)
        if result == False:
            return "Plant Classification error: " + str(e), 500

        [type, confidence] = result
        print(type, confidence)
        # store data in database
        insertImageDataQuery = text('INSERT INTO public."ModelsImages" ("image", "createdAt", "updatedAt", "UserId", "type", "confidence") VALUES (:image, :createdAt, :updatedAt, :UserId, :type, :confidence);')
        current_date = datetime.now()
        imageData = db.session.execute(insertImageDataQuery, {
            "image":imagename,
            "createdAt":current_date,
            "updatedAt":current_date,
            "UserId": tokenData["UserId"],
            "type": type,
            "confidence": confidence
        })
        db.session.commit()

        # send response
        response = jsonify({"type": type, "image": imagename, "confidence": confidence})
        return response, 200
    
    except Exception as e:
        return "Plant Classification error: " + str(e), 500


@app.route("/api/imagesModel/diseaseDetectionAndPlantClassification", methods=["POST"])
def diseaseDetectionAndPlantClassification():
    try:
        tokenData = g.tokenData
        imagename = g.imagename

        # detect disease model
        diseases, resultImageName = diseaseDetectionModel(imagename)
        type = plantClassificationModel(imagename)

        # store data in database
        if len(diseases) < 1:
            diseases.append("Healthy Plant")
        insertImageDataQuery = text('INSERT INTO public."ModelsImages" ("image", "createdAt", "updatedAt", "UserId", "type", "resultImage") VALUES (:image, :createdAt, :updatedAt, :UserId, :type, :resultImage) RETURNING id;')
        current_date = datetime.now()
        imageData = db.session.execute(insertImageDataQuery, {
            "image":imagename,
            "createdAt":current_date,
            "updatedAt":current_date,
            "UserId": tokenData["UserId"],
            "type": type,
            "resultImage": resultImageName
        })
        imageId = imageData.fetchone()[0]
        for disease in diseases:
            insertDiseasesQuery = text('INSERT INTO public."DetectDiseaseResults" ("diseaseType", "createdAt", "updatedAt", "ModelsImageId") VALUES (:diseaseType, :createdAt, :updatedAt, :ModelsImageId)')
            db.session.execute(insertDiseasesQuery, {
                "diseaseType":disease,
                "createdAt":current_date,
                "updatedAt":current_date,
                "ModelsImageId": imageId
            })
        db.session.commit()

        # send response
        response = {"diseases": diseases, "type": type, "image":imagename, "resultImage":resultImageName}
        return jsonify(response), 200
    
    except Exception as e:
        return "Disease Detection And Plant Classification error: " + str(e), 500


@app.route("/api/counting", methods=["POST"])
def counting():
    try:
        tokenData = g.tokenData
        imagename = g.imagename

        # detect disease model
        diseases, resultImageName = diseaseDetectionModel(imagename)
        type = plantClassificationModel(imagename)

        # store data in database
        if len(diseases) < 1:
            diseases.append("Healthy Plant")
        insertImageDataQuery = text('INSERT INTO public."ModelsImages" ("image", "createdAt", "updatedAt", "UserId", "type", "resultImage") VALUES (:image, :createdAt, :updatedAt, :UserId, :type, :resultImage) RETURNING id;')
        current_date = datetime.now()
        imageData = db.session.execute(insertImageDataQuery, {
            "image":imagename,
            "createdAt":current_date,
            "updatedAt":current_date,
            "UserId": tokenData["UserId"],
            "type": type,
            "resultImage": resultImageName
        })
        imageId = imageData.fetchone()[0]
        for disease in diseases:
            insertDiseasesQuery = text('INSERT INTO public."DetectDiseaseResults" ("diseaseType", "createdAt", "updatedAt", "ModelsImageId") VALUES (:diseaseType, :createdAt, :updatedAt, :ModelsImageId)')
            db.session.execute(insertDiseasesQuery, {
                "diseaseType":disease,
                "createdAt":current_date,
                "updatedAt":current_date,
                "ModelsImageId": imageId
            })
        db.session.commit()

        # send response
        response = {"diseases": diseases, "type": type, "image":imagename, "resultImage":resultImageName}
        return jsonify(response), 200
    
    except Exception as e:
        return "Disease Detection And Plant Classification error: " + str(e), 500


@app.route("/api/getMyHistory", methods=["POST"])
def getMyHistoryEndPoint():
    try:
        tokenData = g.tokenData

        # get data from database
        getImagesData = text('SELECT "ModelsImages"."id", "ModelsImages"."image", "ModelsImages"."createdAt", "ModelsImages"."type", "ModelsImages"."confidence", "ModelsImages"."resultImage" FROM public."ModelsImages" WHERE "ModelsImages"."UserId"=:UserId')
        getDiseases = text('SELECT "DetectDiseaseResults"."diseaseType" FROM public."DetectDiseaseResults" WHERE "DetectDiseaseResults"."ModelsImageId"=:ImageId')
        images = db.session.execute(getImagesData, {"UserId": tokenData["UserId"]})
        images = images.mappings().all()
        images = [dict(image) for image in images]
        diseases = []
        for i in range(len(images)):
            diseases = db.session.execute(getDiseases, {"ImageId": images[i]["id"]})
            diseases = diseases.mappings().all()
            diseases = [dict(disease) for disease in diseases]
            images[i]["diseases"] = [disease["diseaseType"] for disease in diseases]
        
        # send response
        return jsonify(images), 200
    
    except Exception as e:
        return "Get My History error: " + str(e), 500


@app.route("/api/deleteFromHistory/<id>", methods=["PUT"])
def deleteFromHistoryEndPoint(id):
    try:
        tokenData = g.tokenData

        # get data from database
        getImageData = text('SELECT "ModelsImages"."image", "ModelsImages"."resultImage" FROM public."ModelsImages" WHERE "ModelsImages"."UserId"=:UserId AND "ModelsImages"."id"=:id')
        image = db.session.execute(getImageData, {"UserId": tokenData["UserId"], "id": id})
        image = image.mappings().all()
        image = [dict(image) for image in image]
        if len(image) < 1:
            return "Image not found", 400
        image = image[0]

        # delete image from database
        deleteImageQuery = text('DELETE FROM public."ModelsImages" WHERE "ModelsImages"."id"=:id')
        db.session.execute(deleteImageQuery, {"id": id})
        db.session.commit()

        # delete image from folder
        removeFile(image["image"])
        if image["resultImage"] is not None:
            removeFile(image["resultImage"])

        # send response
        return "Image deleted", 200
    
    except Exception as e:
        return "Delete From History error: " + str(e), 500


@app.route("/api/getImage/<image>", methods=["POST"])
def getImage(image):
    try:
        if os.path.exists(imagesFolderURL+image):
            return send_from_directory(imagesFolderURL, image, mimetype="image/jpeg")
        else:
            return "File not found", 400
        
    except Exception as e:
        return "Get Image error: " + str(e), 500
