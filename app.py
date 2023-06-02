from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
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


@app.route('/')
def hello_world():
    return 'Hello, World!'
