import os
import asyncio
from fastapi import Depends, FastAPI, BackgroundTasks
from sqlalchemy.orm import Session
import boto3
from botocore.config import Config
from dotenv import load_dotenv

from . import models
from .database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

load_dotenv()

app = FastAPI()

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME = "gazai"
s3_config = Config(
    retries={"max_attempts": 10, "mode": "adaptive"},
    connect_timeout=120,
    read_timeout=300,
)
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name="ap-northeast-1",
    config=s3_config,
)


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/model/{model_id}/download")
def upload_model(
    model_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
):
    lora_model = db.query(models.LoraModel).filter_by(id=model_id).first()
    object_key = lora_model.objectKey

    # Define the background task
    async def download_file_task():
        try:
            await asyncio.to_thread(
                s3.download_file,
                BUCKET_NAME,
                object_key,
                f"/workspace/stable-diffusion-webui/models/Lora/{object_key.split('/')[-1]}",
            )
            print(f"File download completed for model {model_id}")
        except Exception as e:
            print(f"Error downloading file for model {model_id}: {str(e)}")

    # Add the task to background tasks
    background_tasks.add_task(download_file_task)

    return {"message": f"Download for model {model_id} started in the background"}
