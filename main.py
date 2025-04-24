from http.client import HTTPException
from typing import Union
from fastapi import FastAPI, Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def read_root(db=Depends(get_db)):
    return {"message": "Connected to Supabase DB!"}



from fastapi import FastAPI
from supabase import create_client
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend access (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
# Supabase credentials
supabase = create_client(url, key)

@app.get("/meldingen")
def get_meldingen():
    response = supabase.table("Messages").select("*").execute()
    return response.data


@app.get("/meldingen/{melding_id}")
def get_melding_by_id(melding_id: int):
    response = supabase.table("Messages").select("*").eq("id", melding_id).execute()
    
    if response.data and len(response.data) > 0:
        return response.data[0]
    else:
        # Return 404 if not found
        raise HTTPException(status_code=404, detail="Melding not found")
