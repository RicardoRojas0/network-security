import os
import sys
import pandas as pd
import certifi
import pymongo
from dotenv import load_dotenv
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.pipelines.training_pipeline import TrainingPipeline
from src.utils.utils import load_preprocessor
from src.utils.model_estimator import ModelEstimator
from src.constants.training_pipeline import (
    DATA_INGESTION_DATABASE_NAME,
    DATA_INGESTION_COLLECTION_NAME,
)
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run
from starlette.responses import RedirectResponse

ca = certifi.where()

load_dotenv()
mongo_db_url = os.getenv("MONGODB_URI")
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = client[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


templates = Jinja2Templates(directory="./templates")


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_roue():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Model training was successful")
    except Exception as e:
        raise NetworkSecurityException(error_message=e)


@app.get("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        preprocessor = load_preprocessor("models/preprocessor.pkl")
        model = load_preprocessor("models/model.pkl")
        model_estimator = ModelEstimator(preprocessor=preprocessor, model=model)
        print(df.iloc[0])

        y_pred = model_estimator.predict(df)
        print(y_pred)

        df["predicted_column"] = y_pred
        print(df["predicted_column"])

        df.to_csv("predictions/prediction.csv")
        table_html = df.to_html(classes="table table-striped")
        return templates.TemplateResponse(
            "table.html", {"request": request, "table": table_html}
        )
    except Exception as e:
        raise NetworkSecurityException(error_message=e)


if __name__ == "__main__":
    app_run(app=app, host="localhost", port=8000)
