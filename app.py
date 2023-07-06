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

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  #solve the dead kernel problem
HOME = os.getcwd()
os.chdir(HOME)
SOURCE_VIDEO_PATH = f"{HOME}/target.mp4"
os.chdir(HOME+ '\ByteTrack')
print (os.getcwd())

from IPython import display
import ultralytics
# import yolox
# print("yolox.__version__:", yolox.__version__)
# from yolox.tracker.byte_tracker import BYTETracker, STrack
# from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass

import supervision
print("supervision.__version__:", supervision.__version__)
from supervision.draw.color import ColorPalette
# from supervision.geometry.dataclasses import Point
# from supervision.video.dataclasses import VideoInfo
# from supervision.video.source import get_video_frames_generator
# from supervision.video.sink import VideoSink
# from supervision.notebook.utils import show_frame_in_notebook
# from supervision.tools.detections import Detections, BoxAnnotator
# from supervision.tools.line_counter import LineCounter, LineCounterAnnotator

from typing import List
from tqdm import tqdm # to show the progress

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
# app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:mohamed910@localhost/smart_farm"
db = SQLAlchemy(app)
CORS(app)

detectionModel = YOLO('models/disease.pt')
classificationModel = keras.models.load_model('models/plantType.h5', custom_objects={'FixedDropout': keras.layers.Dropout, 'Addons>F1Score': F1Score}, average="")
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

@dataclass(frozen=True) 
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

imagesFolderURL = "files/modelsImages/"
videosFolderURL = "files/modelVideos/"

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# def detections2boxes(detections: Detections) -> np.ndarray:
#     return np.hstack((
#         detections.xyxy,
#         detections.confidence[:, np.newaxis]
#     ))

# # converts List[STrack] into format that can be consumed by match_detections_with_tracks function
# def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
#     return np.array([
#         track.tlbr
#         for track
#         in tracks
#     ], dtype=float)

# # matches our bounding boxes with predictions
# def match_detections_with_tracks(detections: Detections, tracks: List[STrack]) -> Detections:
#     if not np.any(detections.xyxy) or len(tracks) == 0:
#         return np.empty((0,))

#     tracks_boxes = tracks2boxes(tracks=tracks)
#     iou = box_iou_batch(tracks_boxes, detections.xyxy)
#     track2detection = np.argmax(iou, axis=1)
    
#     tracker_ids = [None] * len(detections)
    
#     for tracker_index, detection_index in enumerate(track2detection):
#         if iou[tracker_index, detection_index] != 0:
#             tracker_ids[detection_index] = tracks[tracker_index].track_id

#     return tracker_ids
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------


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

# def countingModel():
#     # settings
#     os.chdir(HOME)
#     MODEL = "models/counting.pt"
#     model = YOLO(MODEL)
#     model.fuse()
#     # dict maping class_id to class_name
#     CLASS_NAMES_DICT = model.model.names
#     # class_ids of interest - car, motorcycle, bus and truck
#     CLASS_ID = [0]
#     # create frame generator
#     generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
#     # create instance of BoxAnnotator
#     box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
#     # acquire first video frame
#     iterator = iter(generator)
#     frame = next(iterator)
#     # model prediction on single frame and conversion to supervision Detections
#     results = model(frame)
#     detections = Detections(
#         xyxy=results[0].boxes.xyxy.cpu().numpy(),
#         confidence=results[0].boxes.conf.cpu().numpy(),
#         class_id=results[0].boxes.cls.cpu().numpy().astype(int)
#     )

#     # format custom labels
#     labels = [
#         f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
#         for _, confidence, class_id, tracker_id
#         in detections
#     ] 
#     # annotate and display frame
#     frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)

#     TARGET_VIDEO_PATH = f"{HOME}/apple_result.mp4"
#     VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

#     # create BYTETracker instance
#     byte_tracker = BYTETracker(BYTETrackerArgs())
#     # create VideoInfo instance
#     video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
#     # create frame generator
#     generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
#     # create instance of BoxAnnotator and LineCounterAnnotator
#     box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=2)
#     unique_tracker_ids = set() #to count the number of apples

#     # open target video file
#     with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
#         # loop over video frames
#         for frame in tqdm(generator, total=video_info.total_frames):
#             # model prediction on single frame and conversion to supervision Detections
#             results = model(frame)
#             detections = Detections(
#                 xyxy=results[0].boxes.xyxy.cpu().numpy(),
#                 confidence=results[0].boxes.conf.cpu().numpy(),
#                 class_id=results[0].boxes.cls.cpu().numpy().astype(int)
#             )
#             # filtering out detections with unwanted classes
#             mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
#             detections.filter(mask=mask, inplace=True)
#             # tracking detections
#             tracks = byte_tracker.update(
#                 output_results=detections2boxes(detections=detections),
#                 img_info=frame.shape,
#                 img_size=frame.shape
#             )
#             tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
#             detections.tracker_id = np.array(tracker_id)
            
            
#             # filtering out detections without trackers
#             mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
#             detections.filter(mask=mask, inplace=True)
            
#             # format custom labels
#             labels = [
#                 f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
#                 for _, confidence, class_id, tracker_id
#                 in detections
#             ]

#             # annotate and display frame
#             frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
#             #line_annotator.annotate(frame=frame, line_counter=line_counter)
#             sink.write_frame(frame)
            
#         for tracker_id in detections.tracker_id:
#             unique_tracker_ids.add(tracker_id)

#     apples_number = len(unique_tracker_ids)
#     # print(f"Number of apples are {apples_number}")
#     return apples_number

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


# @app.route("/api/counting", methods=["POST"])
# def counting():
#     return jsonify({"count":countingModel()}), 200


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
