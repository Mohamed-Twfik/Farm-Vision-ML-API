from flask import Flask, jsonify, request, send_from_directory, g
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
from tensorflow import keras
import numpy as np
from flask_cors import CORS
import os
import base64
# from tensorflow_addons.metrics import F1Score

import yolox
print("yolox.__version__:", yolox.__version__)
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch

import supervision
print("supervision.__version__:", supervision.__version__)
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator

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


def diseaseDetectionModel(imagename, resultImageName):
    try:
        img = Image.open(imagesFolderURL+imagename)
        img = img.resize((640, 640)) # resize to match model input size
        img.save(imagename)

        detectionModel = YOLO('models/disease.pt')
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
        os.rename(imagesFolderURL+"predict/"+imagename, imagesFolderURL+resultImageName)
        os.rmdir(imagesFolderURL+"predict/")

        return True, output
    
    except Exception as e:
        return False, e

def plantClassificationModel(imagename):
    try:
        classificationModel = keras.models.load_model('models/plantType.h5' , custom_objects={'FixedDropout': keras.layers.Dropout, 'Addons>F1Score': F1Score} )
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
            return True, [class_name, confidence]
        else:
            return False, "Error in class name"
    
    except Exception as e:
        return False, e


def diseasePrepareData(imagename, resultImageName):
    # prepare data
    status, result = diseaseDetectionModel(imagename, resultImageName)
    if status == False:
        return jsonify({"message": "Disease detection error: " + str(result)}), 500
    diseases = result

    resultImageBlob = None
    with open(imagesFolderURL+resultImageName, 'rb') as file:
        resultImageBlob = bytes(file.read())

    imageBlob = None
    with open(imagesFolderURL+imagename, 'rb') as file:
        imageBlob = bytes(file.read())

    if len(diseases) < 1:
        diseases.append("Healthy Plant")
    
    type = None
    confidence = None

    return diseases, imageBlob, resultImageBlob, type, confidence

def classificationPrepareData(imagename):
    # prepare data
    status, result = plantClassificationModel(imagename)
    if status == False:
        return jsonify({"message": "Classification error: " + str(result)}), 500
    [type, confidence] = result

    imageBlob = None
    with open(imagesFolderURL+imagename, 'rb') as file:
        imageBlob = bytes(file.read())

    resultImageBlob = None
    diseases = []
    return diseases, imageBlob, resultImageBlob, type, confidence

def diseasesAndClassificationPrepareData(imagename, resultImageName):
    # prepare data
    status, result = diseaseDetectionModel(imagename, resultImageName)
    if status == False:
        return jsonify({"message": "Disease detection error: " + str(result)}), 500
    diseases = result

    status, result = plantClassificationModel(imagename)
    if status == False:
        return jsonify({"message": "Classification error: " + str(result)}), 500
    [type, confidence] = result

    resultImageBlob = None
    with open(imagesFolderURL+resultImageName, 'rb') as file:
        resultImageBlob = bytes(file.read())

    imageBlob = None
    with open(imagesFolderURL+imagename, 'rb') as file:
        imageBlob = bytes(file.read())

    if len(diseases) < 1:
        diseases.append("Healthy Plant")

    return diseases, imageBlob, resultImageBlob, type, confidence


def countingModel(videoname, resultVideoname):
    try:
        # settings
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  #solve the dead kernel problem
        HOME = os.getcwd()
        os.chdir(HOME)
        os.chdir(HOME+ '\ByteTrack')
        print (os.getcwd())

        # import yolox
        # print("yolox.__version__:", yolox.__version__)
        # from yolox.tracker.byte_tracker import BYTETracker, STrack
        # from onemetric.cv.utils.iou import box_iou_batch

        from dataclasses import dataclass
        from typing import List
        from tqdm import tqdm # to show the progress
        import numpy as np
        from ultralytics import YOLO

        @dataclass(frozen=True) 
        class BYTETrackerArgs:
            track_thresh: float = 0.25
            track_buffer: int = 30
            match_thresh: float = 0.8
            aspect_ratio_thresh: float = 3.0
            min_box_area: float = 1.0
            mot20: bool = False

        # import supervision
        # print("supervision.__version__:", supervision.__version__)
        # from supervision.draw.color import ColorPalette
        # from supervision.geometry.dataclasses import Point
        # from supervision.video.dataclasses import VideoInfo
        # from supervision.video.source import get_video_frames_generator
        # from supervision.video.sink import VideoSink
        # from supervision.notebook.utils import show_frame_in_notebook
        # from supervision.tools.detections import Detections, BoxAnnotator
        # from supervision.tools.line_counter import LineCounter, LineCounterAnnotator

        def detections2boxes(detections: Detections) -> np.ndarray:
            return np.hstack((
                detections.xyxy,
                detections.confidence[:, np.newaxis]
            ))

        # converts List[STrack] into format that can be consumed by match_detections_with_tracks function
        def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
            return np.array([
                track.tlbr
                for track
                in tracks
            ], dtype=float)

        # matches our bounding boxes with predictions
        def match_detections_with_tracks(detections: Detections, tracks: List[STrack]) -> Detections:
            if not np.any(detections.xyxy) or len(tracks) == 0:
                return np.empty((0,))

            tracks_boxes = tracks2boxes(tracks=tracks)
            iou = box_iou_batch(tracks_boxes, detections.xyxy)
            track2detection = np.argmax(iou, axis=1)
            
            tracker_ids = [None] * len(detections)
            
            for tracker_index, detection_index in enumerate(track2detection):
                if iou[tracker_index, detection_index] != 0:
                    tracker_ids[detection_index] = tracks[tracker_index].track_id

            return tracker_ids

        # settings
        os.chdir(HOME)
        # SOURCE_VIDEO_PATH = f"{HOME}/target.mp4"
        SOURCE_VIDEO_PATH = HOME + "/" + videosFolderURL + videoname
        MODEL = "models/counting.pt"
        model = YOLO(MODEL)
        model.fuse()
        # dict maping class_id to class_name
        CLASS_NAMES_DICT = model.model.names
        # class_ids of interest - car, motorcycle, bus and truck
        CLASS_ID = [0]
        # create frame generator
        generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
        # create instance of BoxAnnotator
        box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
        # acquire first video frame
        iterator = iter(generator)
        frame = next(iterator)
        # model prediction on single frame and conversion to supervision Detections
        results = model(frame)
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )

        # format custom labels
        labels = [
            f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ] 
        # annotate and display frame
        frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
        
        # TARGET_VIDEO_PATH = f"{HOME}/apple_result.mp4"
        TARGET_VIDEO_PATH = HOME + "/" + videosFolderURL + resultVideoname

        VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

        # create BYTETracker instance
        byte_tracker = BYTETracker(BYTETrackerArgs())
        # create VideoInfo instance
        video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
        # create frame generator
        generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
        # create instance of BoxAnnotator and LineCounterAnnotator
        box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=2)
        unique_tracker_ids = set() #to count the number of apples

        # open target video file
        with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
            # loop over video frames
            for frame in tqdm(generator, total=video_info.total_frames):
                # model prediction on single frame and conversion to supervision Detections
                results = model(frame)
                detections = Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int)
                )
                # filtering out detections with unwanted classes
                mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
                detections.filter(mask=mask, inplace=True)
                # tracking detections
                tracks = byte_tracker.update(
                    output_results=detections2boxes(detections=detections),
                    img_info=frame.shape,
                    img_size=frame.shape
                )
                tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
                detections.tracker_id = np.array(tracker_id)
                
                
                # filtering out detections without trackers
                mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
                detections.filter(mask=mask, inplace=True)
                
                # format custom labels
                labels = [
                    f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                    for _, confidence, class_id, tracker_id
                    in detections
                ]

                # annotate and display frame
                frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
                #line_annotator.annotate(frame=frame, line_counter=line_counter)
                sink.write_frame(frame)
                
            for tracker_id in detections.tracker_id:
                unique_tracker_ids.add(tracker_id)

        apples_number = len(unique_tracker_ids)
        return 200, apples_number

    except Exception as e:
        return 500, e


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
        features = request.form.get('features')
        if features is None:
            return jsonify({"message": "Features is required"}), 400
        
        checkAccessFunc = checkAccess(features, tokenData["UserId"], imagesFolderURL+imagename)
        if not checkAccessFunc:
            return jsonify({"message": "You don't have access to this feature"}), 401

        if "classification" in features and "diseases" in features:
            diseases, imageBlob, resultImageBlob, type, confidence = diseasesAndClassificationPrepareData(imagename, resultImageName)
        elif "classification" in features:
            diseases, imageBlob, resultImageBlob, type, confidence = classificationPrepareData(imagename)
        elif "diseases" in features:
            diseases, imageBlob, resultImageBlob, type, confidence = diseasePrepareData(imagename, resultImageName)
        else:
            return jsonify({"message": "Invalid features"}), 400

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