from ultralytics import YOLO
from PIL import Image
from tensorflow import keras
from tensorflow_addons.metrics import F1Score
import numpy as np
import os

imagesFolderURL = "files/modelsImages/"

def removeFile(url):
    try:
        if os.path.exists(url):
            os.remove(url)
            return True, "File deleted successfully"
        else:
            return True, "File not found"
    except Exception as e:
        return False, "Remove File error: " + str(e)


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
        return 500, result
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

    return 200, [diseases, imageBlob, resultImageBlob, type, confidence]

def classificationPrepareData(imagename):
    # prepare data
    status, result = plantClassificationModel(imagename)
    if status == False:
        return 500, result
    [type, confidence] = result

    imageBlob = None
    with open(imagesFolderURL+imagename, 'rb') as file:
        imageBlob = bytes(file.read())

    resultImageBlob = None
    diseases = []
    return 200, [diseases, imageBlob, resultImageBlob, type, confidence]

def diseasesAndClassificationPrepareData(imagename, resultImageName):
    # prepare data
    status, result = diseaseDetectionModel(imagename, resultImageName)
    if status == False:
        return 500, result
    
    diseases = result

    status, result = plantClassificationModel(imagename)
    if status == False:
        return 500, result
    [type, confidence] = result

    resultImageBlob = None
    with open(imagesFolderURL+resultImageName, 'rb') as file:
        resultImageBlob = bytes(file.read())

    imageBlob = None
    with open(imagesFolderURL+imagename, 'rb') as file:
        imageBlob = bytes(file.read())

    if len(diseases) < 1:
        diseases.append("Healthy Plant")

    return 200, [diseases, imageBlob, resultImageBlob, type, confidence]
