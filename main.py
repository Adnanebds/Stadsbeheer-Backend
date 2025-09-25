from http.client import HTTPException
from typing import Union
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException as FastAPIHTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from supabase import create_client
import os
import magic
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import tempfile
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
SUPABASE_URL = os.getenv("SUPABASE_URL") 
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Security configurations
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_MIME_TYPES = {
    'pdf': 'application/pdf',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'doc': 'application/msword',
    'txt': 'text/plain',
    'xml': ['application/xml', 'text/xml']
}

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="XML Validation API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Supabase setup
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# CORS - more restrictive
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Specific origins only
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only needed methods
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def validate_file_security(file: UploadFile) -> dict:
    """
    Validate file for security issues
    Returns dict with validation result
    """
    if not file.filename:
        raise FastAPIHTTPException(status_code=400, detail="File must have a name")
    
    # Check file size
    if file.size and file.size > MAX_FILE_SIZE:
        raise FastAPIHTTPException(
            status_code=400, 
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE/1024/1024:.1f}MB"
        )
    
    # Get file extension
    file_ext = Path(file.filename).suffix.lower().lstrip('.')
    if not file_ext:
        raise FastAPIHTTPException(status_code=400, detail="File must have an extension")
    
    # Create temporary file to check MIME type
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        content = file.file.read(8192)  # Read first 8KB for MIME detection
        tmp_file.write(content)
        tmp_file.flush()
        
        # Reset file pointer
        file.file.seek(0)
        
        # Check actual MIME type
        detected_mime = magic.from_file(tmp_file.name, mime=True)
        
        # Clean up temp file
        os.unlink(tmp_file.name)
    
    # Validate MIME type matches extension
    allowed_mimes = ALLOWED_MIME_TYPES.get(file_ext, [])
    if isinstance(allowed_mimes, str):
        allowed_mimes = [allowed_mimes]
    
    if not allowed_mimes or detected_mime not in allowed_mimes:
        raise FastAPIHTTPException(
            status_code=400, 
            detail=f"Invalid file type. Extension: {file_ext}, Detected: {detected_mime}"
        )
    
    return {
        "filename": file.filename,
        "extension": file_ext,
        "mime_type": detected_mime,
        "size": file.size
    }

@app.get("/")
@limiter.limit("10/minute")
def read_root(request: Request, db=Depends(get_db)):
    return {"message": "Connected to Supabase DB!", "status": "secure"}

@app.get("/meldingen")
@limiter.limit("30/minute")  # Rate limit API calls
def get_meldingen(request: Request):
    try:
        response = supabase.table("Messages").select("*").limit(100).execute()  # Limit results
        return {"data": response.data, "count": len(response.data)}
    except Exception as e:
        raise FastAPIHTTPException(status_code=500, detail="Database error")

@app.get("/meldingen/{melding_id}")
@limiter.limit("60/minute")
def get_melding_by_id(request: Request, melding_id: int):
    # Basic input validation
    if melding_id <= 0 or melding_id > 999999:  # Reasonable ID range
        raise FastAPIHTTPException(status_code=400, detail="Invalid melding ID")
    
    try:
        response = supabase.table("Messages").select("*").eq("id", melding_id).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]
        else:
            raise FastAPIHTTPException(status_code=404, detail="Melding not found")
    except Exception as e:
        raise FastAPIHTTPException(status_code=500, detail="Database error")

@app.post("/upload/rules")
@limiter.limit("5/minute")  # Stricter limit for file uploads
async def upload_business_rules(
    request: Request, 
    file: UploadFile = File(...)
):
    """
    Upload business rules file with security validation
    """
    # Validate file security
    file_info = validate_file_security(file)
    
    # For now, just validate and return info
    # In next iteration, we'll add actual processing
    return {
        "message": "File validation passed",
        "file_info": file_info,
        "status": "ready_for_processing"
    }

@app.post("/upload/xml")
@limiter.limit("10/minute")
async def upload_xml_message(
    request: Request,
    file: UploadFile = File(...)
):
    """
    Upload XML message file with security validation
    """
    # Validate file security  
    file_info = validate_file_security(file)
    
    # Additional validation for XML files
    if file_info["extension"] not in ["xml", "txt"]:
        raise FastAPIHTTPException(
            status_code=400, 
            detail="Only XML and TXT files allowed for messages"
        )
    
    return {
        "message": "XML file validation passed", 
        "file_info": file_info,
        "status": "ready_for_processing"
    }

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "security_features": [
            "file_validation",
            "mime_type_checking", 
            "rate_limiting",
            "cors_restrictions"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)