from datetime import datetime
from typing import List
from sqlalchemy import Column, DateTime, ForeignKey, String, Enum, Table
from sqlalchemy.orm import declarative_base, relationship, Mapped
from enum import Enum as PyEnum

from sqlalchemy_serializer import SerializerMixin

from .database import Base


training_image_table = Table(
    "TrainingImage",
    Base.metadata,
    Column("loraModelId", String, ForeignKey("LoraModel.id")),
    Column("imageId", String, ForeignKey("Image.id")),
)


class LoraModelStatus(PyEnum):
    PENDING = "PENDING"
    TRAINING = "TRAINING"
    READY = "READY"
    ERROR = "ERROR"


class Image(Base):
    __tablename__ = "Image"

    id = Column(String, primary_key=True)
    objectKey = Column(String)
    fileName = Column(String)
    caption = Column(String, nullable=True)
    userId = Column(String)
    createdAt = Column(DateTime, default=datetime.now)


class LoraModel(Base, SerializerMixin):
    __tablename__ = "LoraModel"

    id = Column(String, primary_key=True)
    name = Column(String)
    title = Column(String, nullable=True)
    fileName = Column(String, nullable=True)
    description = Column(String, nullable=True)
    baseModel = Column(String)
    resolution = Column(String, nullable=True)
    instancePrompt = Column(String, nullable=True)
    classPrompt = Column(String, nullable=True)
    objectKey = Column(String, nullable=True)
    status = Column(
        Enum(LoraModelStatus), default=LoraModelStatus.PENDING, nullable=False
    )

    userId = Column(String)
    createdAt = Column(DateTime, default=datetime.now)


    trainingImages: Mapped[List[Image]] = relationship(secondary=training_image_table)
