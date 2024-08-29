import os
from fastapi import HTTPException
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import QueuePool
import time
import random

load_dotenv()

SQLALCHEMY_DATABASE_URL = os.environ.get("DB_URL")


def create_db_engine(max_retries=5, initial_backoff=1):
    retries = 0
    while retries < max_retries:
        try:
            engine = create_engine(
                SQLALCHEMY_DATABASE_URL,
                connect_args={
                    "connect_timeout": 60  # Increase connection timeout to 60 seconds
                },
                poolclass=QueuePool,
                pool_size=5,  # Adjust based on your needs
                max_overflow=10,
                pool_timeout=30,  # Connection pool timeout
                pool_recycle=1800,  # Recycle connections after 30 minutes
            )
            engine.connect()
            return engine
        except OperationalError as e:
            retries += 1
            if retries == max_retries:
                raise HTTPException(
                    status_code=500, detail="Unable to connect to the database"
                )
            backoff = initial_backoff * (2 ** (retries - 1)) + random.uniform(0, 1)
            print(
                f"Database connection attempt {retries} failed. Retrying in {backoff:.2f} seconds..."
            )
            time.sleep(backoff)


engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
