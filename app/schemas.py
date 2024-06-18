from pydantic import BaseModel, Field
from .models import LoraModelStatus  # Import the enum


class LoraModelCreate(BaseModel):
    name: str = Field(..., max_length=100)
    title: str = Field(..., max_length=100)
    fileName: str = Field(..., max_length=100)
    description: str | None = None
    baseModel: str = Field(..., max_length=100)
    resolution: str | None = None
    instancePrompt: str | None = None
    classPrompt: str | None = None
    trainingImageIds: list[str] = []
    userId: str


class LoraModelRead(BaseModel):
    id: str
    name: str
    description: str | None = None
    baseModel: str
    resolution: str | None = None
    objectKey: str | None = None
    status: LoraModelStatus
    userId: str
    regDataset: str | None = None
    createdAt: str  # Or datetime, depending on your database type
