from flask import Flask, jsonify, request, send_from_directory, g
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from datetime import datetime
from flask_cors import CORS
import os
import base64
# from CountingModel import countingModel
from imagesModels import diseasesAndClassificationPrepareData, classificationPrepareData, diseasePrepareData

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
# app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:mohamed910@localhost/smart_farm"
db = SQLAlchemy(app)
CORS(app)

imagesFolderURL = "files/modelsImages/"
videosFolderURL = "files/modelVideos/"

def removeFile(url):
    try:
        if os.path.exists(url):
            os.remove(url)
            return True, "File deleted successfully"
        else:
            return True, "File not found"
    except Exception as e:
        return False, "Remove File error: " + str(e)

def checkAccess(features, userId, fileURL):
    getUserHaveFreeTrial = text('SELECT "Users"."haveFreeTrial" FROM public."Users" WHERE "Users"."id" = :userId;')
    haveFreeTrialResult = db.session.execute(getUserHaveFreeTrial, {"userId":userId})
    haveFreeTrialResult = haveFreeTrialResult.mappings().all()

    if not haveFreeTrialResult[0]["haveFreeTrial"]:
        getUserFeatures = text('SELECT "Features"."feature" FROM public."Features" INNER JOIN public."UserFeatures" ON ("UserFeatures"."FeatureId" = "Features"."id") WHERE "UserFeatures"."UserId" = :userId;')
        featuresData = db.session.execute(getUserFeatures, {"userId":userId})
        featuresData = featuresData.mappings().all()
        featuresData = [dict(featureData)["feature"] for featureData in featuresData]
        if len(featuresData) == 0:
            return False
        else:
            for feature in features:
                if feature not in featuresData:
                    removeFile(fileURL)
                    return False
            return True
    else:
        return True


@app.before_request
def authentication():
    try:
        auth_token = request.form.get('x-auth-token')
        if auth_token is None:
            return jsonify({"message": "Token is required"}), 401
        
        getTokenQuery = text('SELECT "Tokens"."token", "Tokens"."UserId" FROM public."Tokens" WHERE "Tokens"."token"=:token')
        tokenResult = db.session.execute(getTokenQuery, {"token": auth_token})
        tokenData = tokenResult.mappings().all()
        if not tokenData:
            return jsonify({"message": "Token is invalid"}), 401
        
        getUserQuery = text('SELECT "Users"."premium", "Users"."haveFreeTrial" FROM public."Users" WHERE "Users"."id"=:id')
        userResult = db.session.execute(getUserQuery, {"id": tokenData[0]['UserId']})
        userData = userResult.mappings().all()
        if userData[0]['premium'] or userData[0]['haveFreeTrial']:
            g.tokenData = tokenData[0]
        else:
            return jsonify({"message": "Sorry you can't use our features please subscribe first"}), 401

    except Exception as e:
        return jsonify({"message": "Authentication error: " + str(e)}), 500

@app.before_request
def imageValidation():
    try:
        if request.path.startswith('/api/imagesModels/process'):
            if "image" not in request.files:
                return jsonify({"message": "No file part"}), 400
            image = request.files["image"]
            if image.filename == "":
                return jsonify({"message": "No selected file"}), 400
            
            # save the file with a secure filename
            now = datetime.now()
            timestamp = now.timestamp()
            milliseconds = round(timestamp * 1000)
            imagename = "image-" + str(milliseconds) + ".jpg"
            image.save(imagesFolderURL+imagename)
            g.imagename = imagename
            g.resultImageName = "result-" + imagename

    except Exception as e:
        return jsonify({"message": "Image validation error: " + str(e)}), 500

@app.before_request
def videoValidation():
    try:
        if request.path.startswith('/api/videosModels/process'):
            if "video" not in request.files:
                return jsonify({"message": "No file part"}), 400
            video = request.files["video"]
            if video.filename == "":
                return jsonify({"message": "No selected file"}), 400

            # save the file with a secure filename
            now = datetime.now()
            timestamp = now.timestamp()
            milliseconds = round(timestamp * 1000)
            videoname = "video-" + str(milliseconds) + ".mp4"
            video.save(videosFolderURL+videoname)
            g.videoname = videoname
            g.resultVideoname = "result-" + videoname

    except Exception as e:
        return jsonify({"message": "Video validation error: " + str(e)}), 500

# ---------------------------------------------------------------------------------------

@app.route("/api/imagesModels/process", methods=["POST"])
def imagesModels():
    try:
        tokenData = g.tokenData
        imagename = g.imagename
        resultImageName = g.resultImageName
        features = request.form.getlist('features[]')

        if features is None:
            return jsonify({"message": "Features is required"}), 400
        
        checkAccessFunc = checkAccess(features, tokenData["UserId"], imagesFolderURL+imagename)
        if not checkAccessFunc:
            return jsonify({"message": "You don't have access to this feature"}), 401

        if "classification" in features and "diseases" in features:
            code, result = diseasesAndClassificationPrepareData(imagename, resultImageName)
        elif "classification" in features:
            code, result = classificationPrepareData(imagename)
        elif "diseases" in features:
            code, result = diseasePrepareData(imagename, resultImageName)
        else:
            return jsonify({"message": "Invalid features"}), 400

        [diseases, imageBlob, resultImageBlob, type, confidence] = [None, None, None, None, None]
        if code == 500:
            return jsonify({"message": "image processing error: " + str()}), code
        else:
            [diseases, imageBlob, resultImageBlob, type, confidence] = result
        # store data in database
        insertImageDataQuery = text('INSERT INTO public."ModelsImages" ("image", "createdAt", "updatedAt", "UserId", "resultImage", "type", "confidence") VALUES (:image, :createdAt, :updatedAt, :UserId, :resultImage, :type, :confidence) RETURNING id;')
        current_date = datetime.now()
        imageData = db.session.execute(insertImageDataQuery, {
            "image":imageBlob,
            "createdAt":current_date,
            "updatedAt":current_date,
            "UserId": tokenData["UserId"],
            "resultImage": resultImageBlob,
            "type": type,
            "confidence": confidence
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

        removeFile(imagesFolderURL+imagename)
        if resultImageName != None:
            removeFile(imagesFolderURL+resultImageName)

        image = base64.b64encode(imageBlob).decode('utf-8')
        resultImage = None
        if resultImageBlob != None:
            resultImage = base64.b64encode(resultImageBlob).decode('utf-8')
        response = {"message": "Process success", "diseases": diseases, "image": image, "resultImage": resultImage, "type": type, "confidence": confidence}
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"message": "Processing error: " + str(e)}), 500

@app.route("/api/imagesModels", methods=["GET"])
def getImagesModelsData():
    try:
        tokenData = g.tokenData
        getImagesData = text('SELECT "ModelsImages"."id", "ModelsImages"."image", "ModelsImages"."createdAt", "ModelsImages"."type", "ModelsImages"."confidence", "ModelsImages"."resultImage" FROM public."ModelsImages" WHERE "ModelsImages"."UserId"=:UserId')
        getDiseases = text('SELECT "DetectDiseaseResults"."diseaseType" FROM public."DetectDiseaseResults" WHERE "DetectDiseaseResults"."ModelsImageId"=:ImageId')
        images = db.session.execute(getImagesData, {"UserId": tokenData["UserId"]})
        images = images.mappings().all()
        images = [dict(image) for image in images]
        
        diseases = []
        for i in range(len(images)):
            images[i]["image"] = base64.b64encode(images[i]["image"]).decode('utf-8')
            if images[i]["resultImage"]:
                images[i]["resultImage"] = base64.b64encode(images[i]["resultImage"]).decode('utf-8')
            diseases = db.session.execute(getDiseases, {"ImageId": images[i]["id"]})
            diseases = diseases.mappings().all()
            diseases = [dict(disease) for disease in diseases]
            images[i]["diseases"] = [disease["diseaseType"] for disease in diseases]

        response = {"message": "Get images data success", "data": images}
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"message": "Get images data error: " + str(e)}), 500

@app.route("/api/imagesModels/<id>", methods=["GET"])
def getImagesModelsRow(id):
    try:
        getImagesData = text('SELECT "ModelsImages"."id", "ModelsImages"."image", "ModelsImages"."createdAt", "ModelsImages"."type", "ModelsImages"."confidence", "ModelsImages"."resultImage" FROM public."ModelsImages" WHERE "ModelsImages"."id"=:id')
        getDiseases = text('SELECT "DetectDiseaseResults"."diseaseType" FROM public."DetectDiseaseResults" WHERE "DetectDiseaseResults"."ModelsImageId"=:ImageId')
        images = db.session.execute(getImagesData, {"id": id})
        images = images.mappings().all()
        images = [dict(i) for i in images]
        if len(images) < 1:
            return jsonify({"images": "Image not found or access denied"}), 400
        
        diseases = []
        for i in range(len(images)):
            images[i]["image"] = base64.b64encode(images[i]["image"]).decode('utf-8')
            if images[i]["resultImage"]:
                images[i]["resultImage"] = base64.b64encode(images[i]["resultImage"]).decode('utf-8')
            diseases = db.session.execute(getDiseases, {"ImageId": images[i]["id"]})
            diseases = diseases.mappings().all()
            diseases = [dict(disease) for disease in diseases]
            images[i]["diseases"] = [disease["diseaseType"] for disease in diseases]
        
        image = images[0]
        response = {"message": "Get image data success", "data": image}
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"message": "Get image data error: " + str(e)}), 500

@app.route("/api/imagesModels/<id>", methods=["DELETE"])
def deleteImagesModelsRow(id):
    try:
        tokenData = g.tokenData
        getImageData = text('SELECT "ModelsImages"."image", "ModelsImages"."resultImage" FROM public."ModelsImages" WHERE "ModelsImages"."UserId"=:UserId AND "ModelsImages"."id"=:id')
        image = db.session.execute(getImageData, {"UserId": tokenData["UserId"], "id": id})
        image = image.mappings().all()
        image = [dict(image) for image in image]
        if len(image) < 1:
            return jsonify({"message": "Image not found or access denied"}), 400

        deleteImageQuery = text('DELETE FROM public."ModelsImages" WHERE "ModelsImages"."id"=:id')
        db.session.execute(deleteImageQuery, {"id": id})
        db.session.commit()

        return jsonify({"message": "Delete image data success"}), 200
    
    except Exception as e:
        return jsonify({"message": "Delete image data error: " + str(e)}), 500

# -------------------------------------------------------------------------------------

@app.route("/api/videosModels/process", methods=["POST"])
def videosModels():
    try:
        videoname = g.videoname
        resultVideoname = g.resultVideoname
        tokenData = g.tokenData
        features = request.form.get("features")

        checkPermission = checkAccess(features, tokenData["UserId"], videoname)
        if not checkPermission:
            return jsonify({"message": "Access Denied: not allowed you to access this feature"}), 401

        code, result = countingModel(videoname, resultVideoname)
        if code != 200:
            return jsonify({"message": "Counting model error: " + str(result)}), 500
        count = result

        insertVideoDataQuery = text('INSERT INTO public."ModelsVideos" ("video", "createdAt", "updatedAt", "UserId", "type", "number", "resultVideo") VALUES (:video, :createdAt, :updatedAt, :UserId, :type, :number);')
        current_date = datetime.now()
        videoData = db.session.execute(insertVideoDataQuery, {
            "video":videoname,
            "createdAt":current_date,
            "updatedAt":current_date,
            "UserId": tokenData["UserId"],
            "type": "apple",
            "number": count,
            "resultVideo": resultVideoname
        })
        db.session.commit()

        response = {"message":"Process Success", "number":count, "video": videoname, "resultVideo": resultVideoname}
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"message": "Processing error: " + str(e)}), 500

@app.route("/api/videosModels", methods=["GET"])
def getVideosModelsData():
    try:
        tokenData = g.tokenData
        getVideosData = text('SELECT "ModelsVideos"."id", "ModelsVideos"."video", "ModelsVideos"."createdAt", "ModelsVideos"."type", "ModelsVideos"."number" FROM public."ModelsVideos" WHERE "ModelsVideos"."UserId"=:UserId')
        videos = db.session.execute(getVideosData, {"UserId": tokenData["UserId"]})
        videos = videos.mappings().all()
        videos = [dict(v) for v in videos]

        response = {"message": "Get videos data success", "data": videos}
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"message": "Get videos data error: " + str(e)}), 500

@app.route("/api/videosModels/<id>", methods=["GET"])
def getVideosModelsRow(id):
    try:
        getVideosData = text('SELECT "ModelsVideos"."id", "ModelsVideos"."video", "ModelsVideos"."createdAt", "ModelsVideos"."type", "ModelsVideos"."number", "ModelsVideos"."resultVideo" FROM public."ModelsVideos" WHERE "ModelsVideos"."id"=:id')
        videos = db.session.execute(getVideosData, {"id": id})
        videos = videos.mappings().all()
        videos = [dict(video) for video in videos]
        if len(videos) < 1:
            return jsonify({"message": "Video not found or access denied"}), 400
        
        video = videos[0]
        response = {"message": "Get video data success", "data": video}
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"message": "Get video data error: " + str(e)}), 500

@app.route("/api/videosModels/<id>", methods=["DELETE"])
def deleteVideosModelsRow(id):
    try:
        tokenData = g.tokenData
        getVideosData = text('SELECT "ModelsVideos"."id", "ModelsVideos"."video", "ModelsVideos"."createdAt", "ModelsVideos"."type", "ModelsVideos"."number" FROM public."ModelsVideos" WHERE "ModelsVideos"."UserId"=:UserId AND "ModelsVideos"."id"=:id')
        video = db.session.execute(getVideosData, {"UserId": tokenData["UserId"], "id": id})
        video = video.mappings().all()
        video = [dict(image) for image in video]
        if len(video) < 1:
            return jsonify({"message": "Video not found or access denied"}), 400

        deleteVideoQuery = text('DELETE FROM public."ModelsVideos" WHERE "ModelsVideos"."id"=:id')
        db.session.execute(deleteVideoQuery, {"id": id})
        db.session.commit()

        removeFile(videosFolderURL + video["video"])
        if video["resultVideo"] is not None:
            removeFile(videosFolderURL + video["resultVideo"])

        return jsonify({"message": "Delete video data success"}), 200
    
    except Exception as e:
        return jsonify({"message": "Delete video data error: " + str(e)}), 500

@app.route("/api/getVideo/<video>", methods=["GET"])
def getVideo(video):
    try:
        if os.path.exists(videosFolderURL+video):
            return send_from_directory(videosFolderURL, video, mimetype="video/mp4")
        else:
            return jsonify({"message": "File not found"}), 400
        
    except Exception as e:
        return jsonify({"message": "Get Video error: " + str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)