import sys
import os

import certifi

ca = certifi.where()

from dotenv import load_dotenv

load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL")
print(mongo_db_url)
import pymongo
from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object

from networksecurity.utils.ml_utils.utils import NetworkModel

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

from networksecurity.constant.training_constants import COLLECTIONS
from networksecurity.constant.training_constants import DATABASE

database = client[DATABASE]
collection = database[COLLECTIONS]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="./templates")


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.predict()
        return Response("Training is successful")
    except Exception as e:
        raise CustomException(e, sys)


from fastapi.responses import JSONResponse

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        # 1️⃣ Read uploaded CSV
        df = pd.read_csv(file.file)

        # Remove target column if accidentally sent
        if "result" in df.columns:
            df = df.drop(columns=["result"])

        # 2️⃣ Load model artifacts
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")

        network_model = NetworkModel(
            preprocessor=preprocessor,
            model=final_model
        )

        # 3️⃣ Predict
        y_pred = network_model.predict(df)

        # Attach predictions
        df["predicted_column"] = y_pred

        # 4️⃣ Save output CSV (safe)
        output_dir = os.path.join(os.getcwd(), "prediction_output")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "output.csv")
        df.to_csv(output_path, index=False)

        # 5️⃣ Generate HTML table (for browser usage if needed)
        table_html = df.to_html(classes="table table-striped", index=False)

        # 6️⃣ Return BOTH JSON + HTML
        return JSONResponse(
            content={
                "filename": file.filename,
                "saved_to": output_path,
                "rows": len(df),
                "predictions": df.to_dict(orient="records"),
                "html_table": table_html
            }
        )

    except Exception as e:
        raise CustomException(e, sys)




if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)
