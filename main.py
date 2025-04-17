from typing import Union

from fastapi import FastAPI
from fastapi import FastAPI, Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


DATABASE_URL = "postgresql://postgres:1&5819tGw/8^@db.yqlpqgsoynbtvhwprjyt.supabase.co:5432/postgres"  # Replace with your Supabase details

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/omar")
def read_root():
    return {"Hoi": "Omar"}


@app.get("/tarik")
def read_root():
    return {"Hoi": "Tarik"}


@app.get("/adnane")
def read_root():
    return {"Hoi": "adanane"}

@app.get("/")
def read_root(db=Depends(get_db)):
    return {"message": "Connected to Supabase DB!"}
