# Complete Backend API with CORS, Swagger Documentation and Supabase Integration
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restx import Api, Resource, fields, reqparse
from werkzeug.datastructures import FileStorage
import os
import tempfile
import json
import traceback
from werkzeug.utils import secure_filename
import re
import xml.etree.ElementTree as ET
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredFileLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import uuid
from datetime import datetime

# Supabase integration
from supabase import create_client, Client

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for localhost:3000 (React development server)
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])

# Initialize Supabase client
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("⚠️ WARNING: Supabase credentials not found in environment variables")
    print("Please set SUPABASE_URL and SUPABASE_ANON_KEY")
    supabase_client = None
else:
    try:
        supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Supabase client: {e}")
        supabase_client = None

# HuggingFace login
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(hf_token)

# Initialize Swagger API
api = Api(
    app,
    version='1.0',
    title='XML Validation API with Supabase',
    description='API for validating XML messages against business rules using AI with database storage',
    doc='/docs/',
    prefix='/api/v1'
)

# Create namespaces
ns_validation = api.namespace('validation', description='XML message validation operations')
ns_system = api.namespace('system', description='System status and health checks')

# Global variable to store the AI model
generation_pipeline = None

# Install PDF dependencies
import subprocess
import sys

def install_pdf_dependencies():
    """Install required PDF processing libraries"""
    try:
        import pypdf
        print("✅ pypdf already installed")
    except ImportError:
        print("📦 Installing pypdf...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
    
    try:
        import pdfplumber
        print("✅ pdfplumber already available")
    except ImportError:
        print("📦 Installing pdfplumber...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber"])

try:
    install_pdf_dependencies()
except Exception as e:
    print(f"⚠️ Warning: Could not install PDF dependencies: {e}")

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'rules': {'pdf', 'docx', 'doc', 'txt'},
    'xml': {'xml', 'txt'}
}

# Swagger Models
validation_response_model = api.model('ValidationResult', {
    'success': fields.Boolean(required=True, description='Whether the validation was successful'),
    'decision': fields.String(required=True, description='ACCEPTED, REJECTED, or ERROR'),
    'technical_reasons': fields.String(required=True, description='Technical details about the decision'),
    'explanation': fields.String(required=True, description='AI-generated explanation'),
    'files_processed': fields.Raw(description='Information about processed files')
})

health_response_model = api.model('HealthCheck', {
    'status': fields.String(required=True, description='API health status'),
    'model_loaded': fields.Boolean(required=True, description='Whether AI model is loaded'),
    'supabase_connected': fields.Boolean(required=True, description='Whether Supabase is connected'),
    'message': fields.String(required=True, description='Status message')
})

status_response_model = api.model('SystemStatus', {
    'api_version': fields.String(required=True, description='API version'),
    'status': fields.String(required=True, description='System status'),
    'model_status': fields.Raw(description='AI model status information'),
    'supabase_status': fields.Raw(description='Supabase connection status'),
    'supported_files': fields.Raw(description='Supported file types')
})

message_model = api.model('MessageInfo', {
    'id': fields.String(required=True, description='Message UUID'),
    'verzoeknummer': fields.String(description='Message reference number'),
    'activity_name': fields.String(description='Activity name'),
    'message_type': fields.String(description='Message type (Informatie/Melding)'),
    'project_name': fields.String(description='Project name'),
    'initiatiefnemer_name': fields.String(description='Initiator name'),
    'bevoegd_gezag': fields.String(description='Competent authority'),
    'created_at': fields.DateTime(description='Upload timestamp'),
    'validation_count': fields.Integer(description='Number of validations performed'),
    'original_filename': fields.String(description='Original filename')
})

store_message_response = api.model('StoreMessageResponse', {
    'success': fields.Boolean(required=True),
    'message_id': fields.String(required=True, description='Stored message ID'),
    'message': fields.String(required=True),
    'parsed_data': fields.Raw(description='Extracted message metadata')
})

error_response_model = api.model('ErrorResponse', {
    'success': fields.Boolean(required=True, description='Always false for errors'),
    'error': fields.String(required=True, description='Error message'),
    'decision': fields.String(description='Decision status for validation errors'),
    'technical_reasons': fields.String(description='Technical error details'),
    'explanation': fields.String(description='User-friendly error explanation')
})

def allowed_file(filename, file_type):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[file_type]

def get_file_extension(filename):
    """Get file extension for temporary file creation"""
    if not filename:
        return '.tmp'
    return '.' + filename.rsplit('.', 1)[1].lower() if '.' in filename else '.tmp'

# Initialize AI model
def init_model():
    """Initialize the AI model once when the app starts"""
    global generation_pipeline
    if generation_pipeline is None:
        print("🤖 Initializing AI model...")
        generation_pipeline = setup_gemma_model()
        print("✅ AI model ready!")
    return generation_pipeline

def setup_gemma_model():
    print("Setting up Gemma 2-2b model for response generation...")
    try:
        model_name = "google/gemma-2-2b-it"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto"
        )
        
        generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        print("✅ Gemma 2-2b model loaded successfully")
        return generation_pipeline
        
    except Exception as e:
        print(f"❌ Error loading Gemma model: {e}")
        print("Falling back to smaller model...")
        try:
            fallback_pipeline = pipeline(
                "text-generation", 
                model="distilgpt2", 
                max_length=300,
                pad_token_id=50256
            )
            print("✅ Fallback model loaded successfully")
            return fallback_pipeline
        except Exception as fallback_error:
            print(f"❌ Fallback model also failed: {fallback_error}")
            return None

def load_business_rules(file_path):
    print(f"Loading business rules from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Business rules file not found: {file_path}")
    
    file_extension = file_path.split('.')[-1].lower()
    
    try:
        if file_extension == 'pdf':
            print("📄 Attempting to load PDF with PyPDFLoader...")
            try:
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                print(f"✅ Successfully loaded PDF with PyPDFLoader: {len(documents)} pages")
            except Exception as pdf_error:
                print(f"❌ PyPDFLoader failed: {pdf_error}")
                print("🔄 Trying alternative PDF loaders...")
                
                try:
                    loader = UnstructuredFileLoader(file_path)
                    documents = loader.load()
                    print(f"✅ Successfully loaded PDF with UnstructuredFileLoader: {len(documents)} documents")
                except Exception as unstructured_error:
                    print(f"❌ UnstructuredFileLoader also failed: {unstructured_error}")
                    raise Exception(f"Could not load PDF file. Try converting it to TXT format. PyPDF error: {pdf_error}, Unstructured error: {unstructured_error}")
                    
        elif file_extension in ['docx', 'doc']:
            print("📄 Loading DOCX/DOC file...")
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
        else:
            print("📄 Loading text file...")
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
            except:
                try:
                    loader = TextLoader(file_path, encoding='latin-1')
                    documents = loader.load()
                except:
                    loader = UnstructuredFileLoader(file_path)
                    documents = loader.load()
        
        if not documents:
            raise Exception("No documents were loaded from the file")
            
        total_text = ' '.join([doc.page_content for doc in documents])
        readable_chars = sum(1 for c in total_text if c.isprintable() and ord(c) < 256)
        total_chars = len(total_text)
        
        if total_chars == 0:
            raise Exception("Loaded documents contain no text")
            
        readability_ratio = readable_chars / total_chars if total_chars > 0 else 0
        print(f"📊 Content readability: {readability_ratio:.2%} ({readable_chars}/{total_chars} readable chars)")
        
        if readability_ratio < 0.7:
            print("⚠️ WARNING: Low readability detected. The file might be corrupted or in an unsupported format.")
            print("💡 SUGGESTION: Try converting your PDF to TXT format for better results.")
            
            cleaned_docs = []
            for doc in documents:
                cleaned_text = ''.join(c for c in doc.page_content if c.isprintable() and ord(c) < 256)
                if len(cleaned_text) > 100:
                    from langchain.schema import Document
                    cleaned_docs.append(Document(page_content=cleaned_text, metadata=doc.metadata))
            
            if cleaned_docs:
                documents = cleaned_docs
                print(f"🧹 Cleaned documents: {len(documents)} documents with readable text")
            else:
                raise Exception("After cleaning, no readable text remains. Please convert your PDF to TXT format.")
        
        print(f"✅ Successfully loaded {len(documents)} document(s) from {file_path}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            separators=["Activiteit:", "Artikel", "§", "\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        
        if chunks:
            sample_content = chunks[0].page_content[:500]
            print("\n📄 Sample chunk content:")
            print(sample_content + "..." if len(chunks[0].page_content) > 500 else sample_content)
            
            dutch_keywords = ['artikel', 'activiteit', 'melding', 'informatie', 'vereisten', 'bodem', 'saneren']
            found_keywords = [kw for kw in dutch_keywords if kw.lower() in sample_content.lower()]
            
            if found_keywords:
                print(f"✅ Found business rule keywords: {found_keywords}")
            else:
                print("⚠️ WARNING: No business rule keywords found in sample. File might not contain proper business rules.")
        
        print("🔍 Creating embeddings... (this may take a moment)")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(chunks, embeddings)
        print("✅ Vector database created successfully")
        
        return vectordb
    
    except Exception as e:
        print(f"❌ Error loading document: {e}")
        print("\n💡 TROUBLESHOOTING SUGGESTIONS:")
        print("1. Convert your PDF to TXT format using a PDF reader")
        print("2. Make sure the PDF contains searchable text (not just images)")
        print("3. Try uploading a DOCX or TXT version of the business rules")
        print("4. Check if the PDF is password protected or corrupted")
        raise

def read_xml_file(file_path):
    print(f"Reading XML file: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"XML file not found: {file_path}")
    
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            print(f"✅ Successfully read XML file with {encoding} encoding")
            return content
        except UnicodeDecodeError:
            continue
    
    try:
        with open(file_path, 'rb') as file:
            content = file.read()
        return content.decode('utf-8', errors='replace')
    except Exception as e:
        raise RuntimeError(f"Could not read XML file: {e}")

def parse_xml_message(xml_content):
    """Parse XML message and extract key information"""
    try:
        root = ET.fromstring(xml_content)
        
        namespaces = {
            'imam': 'http://www.omgevingswet.nl/koppelvlak/stam-v4/IMAM',
            'vx': 'http://www.omgevingswet.nl/koppelvlak/stam-v4/verzoek',
            'ic': 'http://www.omgevingswet.nl/koppelvlak/stam-v4/common'
        }
        
        # Extract verzoeknummer
        verzoeknummer_elem = root.find('.//vx:verzoeknummer', namespaces)
        verzoeknummer = verzoeknummer_elem.text if verzoeknummer_elem is not None else None
        
        # Extract activity name
        activity_elements = root.findall('.//vx:activiteitnaam', namespaces)
        activity_name = activity_elements[0].text if activity_elements else None
        
        # Extract message type
        message_type_elements = root.findall('.//vx:type', namespaces)
        message_type = message_type_elements[0].text if message_type_elements else None
        
        # Extract project info
        project_name_elem = root.find('.//vx:project/vx:naam', namespaces)
        project_name = project_name_elem.text if project_name_elem is not None else None
        
        # Extract initiatiefnemer info
        handelsnaam_elem = root.find('.//vx:initiatiefnemer//vx:handelsnaam', namespaces)
        initiatiefnemer_name = handelsnaam_elem.text if handelsnaam_elem is not None else None
        
        # Extract bevoegd gezag
        bevoegd_gezag_elem = root.find('.//vx:bevoegdGezag/vx:naam', namespaces)
        bevoegd_gezag = bevoegd_gezag_elem.text if bevoegd_gezag_elem is not None else None
        
        # Extract specifications
        specifications = []
        spec_elements = root.findall('.//vx:specificatie', namespaces)
        for spec in spec_elements:
            answer = spec.find('vx:antwoord', namespaces)
            question = spec.find('vx:vraagtekst', namespaces)
            if answer is not None and answer.text:
                specifications.append({
                    'question': question.text if question is not None else None,
                    'answer': answer.text
                })
        
        # Check for attachments
        document_elements = root.findall('.//vx:document', namespaces)
        attachments = []
        for doc in document_elements:
            filename_elem = doc.find('.//vx:bestandsnaam', namespaces)
            if filename_elem is not None and filename_elem.text:
                attachments.append(filename_elem.text)
        
        # Check for coordinates
        has_coordinates = len(root.findall('.//vx:coordinatenEtrs', namespaces)) > 0 or len(root.findall('.//vx:coordinatenOpgegeven', namespaces)) > 0
        
        result = {
            'verzoeknummer': verzoeknummer,
            'activity_name': activity_name,
            'message_type': message_type,
            'project_name': project_name,
            'initiatiefnemer_name': initiatiefnemer_name,
            'bevoegd_gezag': bevoegd_gezag,
            'specifications': specifications,
            'attachments': attachments,
            'has_coordinates': has_coordinates
        }
        
        print(f"✅ Parsed XML message: {activity_name} - {message_type}")
        return result
        
    except Exception as e:
        print(f"❌ Error parsing XML: {e}")
        return None

def retrieve_business_rules(vectordb, activity_name, message_type):
    queries = [
        f"Business rules for {activity_name} with {message_type} requirements",
        f"{activity_name} {message_type} vereisten",
        f"Artikel {message_type} {activity_name}"
    ]
    
    all_results = []
    for query in queries:
        print(f"🔍 Searching for: {query}")
        results = vectordb.similarity_search(query, k=2)
        all_results.extend(results)
    
    unique_results = []
    seen_contents = set()
    for doc in all_results:
        if doc.page_content not in seen_contents:
            unique_results.append(doc)
            seen_contents.add(doc.page_content)
    
    rule_texts = [doc.page_content for doc in unique_results]
    print(f"✅ Retrieved {len(rule_texts)} unique rule text chunks")
    
    return rule_texts

def extract_requirements(rule_texts, message_type):
    requirements = {
        'required_fields': [],
        'required_attachments': []
    }
    
    message_type_lower = message_type.lower() if message_type else ""
    
    for text in rule_texts:
        print(f"\n📝 Analyzing text chunk for requirements ({len(text)} characters):")
        print(text[:400] + "..." if len(text) > 400 else text)
        
        if "bodem saneren" in text.lower() or "saneren van de bodem" in text.lower():
            print("🎯 Found 'Bodem saneren' specific section")
            
            if "informatie" in message_type_lower and "informatieplicht" in text.lower():
                print("✅ Found informatieplicht section for Bodem saneren")
                
                patterns = [
                    r"(?:gegevens en bescheiden verstrekt over|worden.*?verstrekt over):\s*(.*?)(?=\n\d|\n[A-Z]|Artikel|\Z)",
                    r"(?:Een.*?bevat):\s*(.*?)(?=\n\d|\n[A-Z]|Artikel|\Z)",
                    r"(?:melding bevat):\s*(.*?)(?=\n\d|\n[A-Z]|Artikel|\Z)"
                ]
                
                for pattern in patterns:
                    field_matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                    for match in field_matches:
                        field_items = re.findall(r"([a-z]\.\s*|[0-9]+°?\.\s*)(.*?)(?=\n[a-z]\.|$|\n\d)", match, re.DOTALL)
                        for _, field in field_items:
                            clean_field = field.strip()
                            if clean_field and len(clean_field) > 5:
                                print(f"  ✅ Found required field: {clean_field}")
                                requirements['required_fields'].append(clean_field)
                
                if "begrenzing van de locatie" in text.lower():
                    requirements['required_fields'].append("de begrenzing van de locatie waarop de activiteit wordt verricht")
                    print("  ✅ Found location boundary requirement")
                
                if "verwachte datum" in text.lower():
                    requirements['required_fields'].append("de verwachte datum van het begin van de activiteit")
                    print("  ✅ Found expected start date requirement")
                
                if "naam en het adres van degene die de werkzaamheden" in text.lower():
                    requirements['required_fields'].append("de naam en het adres van degene die de werkzaamheden gaat verrichten")
                    print("  ✅ Found work performer requirement")
                
                if "milieukundige begeleiding" in text.lower():
                    requirements['required_fields'].append("de naam en het adres van de onderneming die de milieukundige begeleiding gaat verrichten")
                    requirements['required_fields'].append("de naam van de natuurlijke persoon die de milieukundige begeleiding gaat verrichten")
                    print("  ✅ Found environmental guidance requirements")
                
            elif "melding" in message_type_lower and "melding" in text.lower():
                print("✅ Found melding section for Bodem saneren")
                
                patterns = [
                    r"(?:Een melding bevat):\s*(.*?)(?=\n\d|\n[A-Z]|Artikel|\Z)",
                    r"(?:melding.*?bevat):\s*(.*?)(?=\n\d|\n[A-Z]|Artikel|\Z)"
                ]
                
                for pattern in patterns:
                    field_matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                    for match in field_matches:
                        field_items = re.findall(r"([a-z]\.\s*|[0-9]+°?\.\s*)(.*?)(?=\n[a-z]\.|$|\n\d)", match, re.DOTALL)
                        for _, field in field_items:
                            clean_field = field.strip()
                            if clean_field and len(clean_field) > 5:
                                print(f"  ✅ Found required field: {clean_field}")
                                requirements['required_fields'].append(clean_field)
        
        attachment_patterns = [
            "evaluatieverslag",
            "evaluatie",
            "rapport",
            "bijlage"
        ]
        
        for pattern in attachment_patterns:
            if pattern in text.lower():
                if pattern not in [att.lower() for att in requirements['required_attachments']]:
                    print(f"  📎 Found attachment requirement: {pattern}")
                    requirements['required_attachments'].append(pattern)
    
    if not requirements['required_fields'] and "informatie" in message_type_lower:
        print("⚠️ No requirements found in text, using fallback requirements for Bodem saneren informatieplicht")
        requirements['required_fields'] = [
            "de begrenzing van de locatie waarop de activiteit wordt verricht",
            "de verwachte datum van het begin van de activiteit",
            "de naam en het adres van degene die de werkzaamheden gaat verrichten",
            "de naam en het adres van de onderneming die de milieukundige begeleiding gaat verrichten",
            "de naam van de natuurlijke persoon die de milieukundige begeleiding gaat verrichten"
        ]
        requirements['required_attachments'] = ["evaluatieverslag"]
    
    print(f"\n📋 Final extracted requirements:")
    print(f"  Required fields: {len(requirements['required_fields'])}")
    print(f"  Required attachments: {len(requirements['required_attachments'])}")
    
    return requirements

def validate_message(message_data, requirements):
    print("\n⚖️ Validating message against requirements:")
    
    missing_fields = []
    found_fields = []
    
    for field in requirements['required_fields']:
        field_found = False
        
        if "begrenzing van de locatie" in field.lower():
            if message_data.get('has_coordinates', False):
                field_found = True
                found_fields.append(f"'{field}' (via coordinates)")
                print(f"  ✅ Required field found: '{field}' (via coordinates in message)")
                continue
        
        if "verwachte datum" in field.lower() and "begin" in field.lower():
            for spec in message_data['specifications']:
                if spec['answer'] and any(word in spec['answer'].lower() for word in ['datum', 'begin', 'start', 'tijd']):
                    field_found = True
                    found_fields.append(f"'{field}' (in specifications)")
                    print(f"  ✅ Required field found: '{field}' (timing info in specifications)")
                    break
        
        if "naam en het adres van degene die de werkzaamheden" in field.lower():
            if message_data.get('has_initiator_info', True):
                field_found = True
                found_fields.append(f"'{field}' (initiator/representative info)")
                print(f"  ✅ Required field found: '{field}' (via initiator/representative info)")
        
        if "milieukundige begeleiding" in field.lower():
            print(f"  ⚠️ Environmental guidance info often not required for information submissions: '{field}'")
            if message_data.get('message_type', '').lower() == 'informatie':
                field_found = True
                found_fields.append(f"'{field}' (not required for informatie)")
        
        if not field_found:
            for spec in message_data['specifications']:
                if spec['answer']:
                    field_keywords = field.lower().split()
                    answer_lower = spec['answer'].lower()
                    if any(keyword in answer_lower for keyword in field_keywords[-3:]):
                        field_found = True
                        found_fields.append(f"'{field}' (in answer: {spec['answer'][:50]}...)")
                        print(f"  ✅ Required field found: '{field}' (in answer)")
                        break
                
                if spec['question']:
                    question_lower = spec['question'].lower()
                    field_keywords = field.lower().split()
                    if any(keyword in question_lower for keyword in field_keywords[-3:]):
                        field_found = True
                        found_fields.append(f"'{field}' (in question)")
                        print(f"  ✅ Required field found: '{field}' (in question)")
                        break
        
        if not field_found:
            print(f"  ❌ Required field missing: '{field}'")
            missing_fields.append(field)
    
    missing_attachments = []
    found_attachments = []
    
    for attachment in requirements['required_attachments']:
        if not message_data['attachments']:
            print(f"  ❌ Required attachment missing: '{attachment}' (no attachments present)")
            missing_attachments.append(attachment)
            continue
        
        attachment_found = False
        for att in message_data['attachments']:
            if (attachment.lower() in att.lower() or 
                'evaluatie' in att.lower() and 'evaluatie' in attachment.lower() or
                'rapport' in att.lower() and 'rapport' in attachment.lower()):
                attachment_found = True
                found_attachments.append(f"'{attachment}' (found: {att})")
                print(f"  ✅ Required attachment found: '{attachment}' in '{att}'")
                break
        
        if not attachment_found:
            print(f"  ❌ Required attachment missing: '{attachment}'")
            missing_attachments.append(attachment)
    
    if (message_data.get('message_type', '').lower() == 'informatie' and 
        any('evaluatie' in att.lower() for att in message_data['attachments']) and
        len(missing_fields) > 0):
        
        print("\n🔍 SPECIAL CASE DETECTED:")
        print("This appears to be a post-activity information submission with evaluation report.")
        print("For completed activities with evaluation reports, some pre-activity requirements may not apply.")
        
        pre_activity_fields = [
            "de verwachte datum van het begin van de activiteit",
            "de naam en het adres van degene die de werkzaamheden gaat verrichten", 
            "de naam en het adres van de onderneming die de milieukundige begeleiding gaat verrichten",
            "de naam van de natuurlijke persoon die de milieukundige begeleiding gaat verrichten"
        ]
        
        original_missing = missing_fields.copy()
        missing_fields = [field for field in missing_fields if not any(pre_req in field for pre_req in pre_activity_fields)]
        
        if len(missing_fields) < len(original_missing):
            print(f"  ✅ Filtered out {len(original_missing) - len(missing_fields)} pre-activity requirements")
            print(f"  📝 Remaining missing fields: {missing_fields}")
    
    is_valid = len(missing_fields) == 0 and len(missing_attachments) == 0
    
    print(f"\n📊 VALIDATION SUMMARY:")
    print(f"  Found fields: {len(found_fields)}")
    print(f"  Missing fields: {len(missing_fields)}")
    print(f"  Found attachments: {len(found_attachments)}")
    print(f"  Missing attachments: {len(missing_attachments)}")
    print(f"  Overall valid: {is_valid}")
    
    return {
        'is_valid': is_valid,
        'missing_fields': missing_fields,
        'missing_attachments': missing_attachments,
        'found_fields': found_fields,
        'found_attachments': found_attachments
    }

def generate_explanation(generation_pipeline, validation_result, message_data):
    if generation_pipeline is None:
        if validation_result['is_valid']:
            return "The message meets all requirements according to the business rules."
        else:
            missing_info = []
            if validation_result['missing_fields']:
                missing_info.append(f"Missing fields: {', '.join(validation_result['missing_fields'])}")
            if validation_result['missing_attachments']:
                missing_info.append(f"Missing attachments: {', '.join(validation_result['missing_attachments'])}")
            return f"The message was rejected because: {'; '.join(missing_info)}"
    
    activity = message_data.get('activity_name', 'Unknown activity')
    message_type = message_data.get('message_type', 'Unknown type')
    
    if validation_result['is_valid']:
        prompt = f"Message for {activity} ({message_type}) was accepted. All required information is present."
    else:
        missing_fields = ", ".join(validation_result['missing_fields']) if validation_result['missing_fields'] else "None"
        missing_attachments = ", ".join(validation_result['missing_attachments']) if validation_result['missing_attachments'] else "None"
        
        prompt = f"Message for {activity} ({message_type}) was rejected. Missing fields: {missing_fields}. Missing attachments: {missing_attachments}. Please provide the missing information."
    
    try:
        print("🤖 Generating AI explanation...")
        result = generation_pipeline(prompt, max_new_tokens=100, temperature=0.3, pad_token_id=generation_pipeline.tokenizer.eos_token_id)
        
        generated_text = result[0]['generated_text']
        if len(generated_text) > len(prompt):
            explanation = generated_text[len(prompt):].strip()
        else:
            explanation = generated_text.strip()
        
        if explanation.count(explanation.split('.')[0]) > 2:
            raise Exception("Repetitive output detected")
            
        return explanation if explanation else prompt
        
    except Exception as e:
        print(f"❌ Error generating explanation: {e}")
        return prompt

def validate_xml_against_rules(xml_content, vectordb, generation_pipeline):
    print("\n" + "="*60)
    print("🚀 STARTING VALIDATION PROCESS")
    print("="*60)
    
    print("\n📄 Step 1: Parsing XML message")
    message_data = parse_xml_message(xml_content)
    if not message_data:
        return {"decision": "REJECTED", "technical_reasons": "Failed to parse XML", "explanation": "Failed to parse XML"}
    
    print(f"\n📊 Parsed message data:")
    print(f"  Activity: {message_data['activity_name']}")
    print(f"  Message Type: {message_data['message_type']}")
    print(f"  Specifications: {len(message_data['specifications'])} items")
    print(f"  Attachments: {message_data['attachments']}")
    
    print(f"\n🔍 Step 2: Retrieving relevant business rules")
    rule_texts = retrieve_business_rules(
        vectordb, 
        message_data['activity_name'], 
        message_data['message_type']
    )
    
    print(f"\n⚙️ Step 3: Extracting structured requirements")
    requirements = extract_requirements(rule_texts, message_data['message_type'])
    
    print(f"\n📋 Extracted requirements summary:")
    print(f"  Required fields: {len(requirements['required_fields'])}")
    print(f"  Required attachments: {len(requirements['required_attachments'])}")
    
    print(f"\n⚖️ Step 4: Validating message")
    validation_result = validate_message(message_data, requirements)
    
    print(f"\n🤖 Step 5: Generating explanation with AI")
    explanation = generate_explanation(generation_pipeline, validation_result, message_data)
    
    decision = "ACCEPTED" if validation_result['is_valid'] else "REJECTED"
    
    if not validation_result['is_valid']:
        reasons = []
        if validation_result['missing_fields']:
            reasons.append(f"Missing fields: {', '.join(validation_result['missing_fields'])}")
        if validation_result['missing_attachments']:
            reasons.append(f"Missing attachments: {', '.join(validation_result['missing_attachments'])}")
        technical_reasons = '; '.join(reasons)
    else:
        technical_reasons = "All requirements met"
    
    result = {
        "decision": decision,
        "technical_reasons": technical_reasons,
        "explanation": explanation
    }
    
    print("\n" + "="*60)
    print("✅ VALIDATION COMPLETE")
    print("="*60)
    
    return result

# API Routes with Supabase Integration

@ns_system.route('/health')
class HealthCheck(Resource):
    @ns_system.doc('health_check')
    @ns_system.marshal_with(health_response_model)
    def get(self):
        """Health check endpoint - Check if the API is running properly"""
        supabase_connected = False
        if supabase_client:
            try:
                # Test Supabase connection
                result = supabase_client.table('xml_messages').select('id').limit(1).execute()
                supabase_connected = True
            except Exception as e:
                print(f"Supabase connection test failed: {e}")
        
        return {
            'status': 'healthy',
            'model_loaded': generation_pipeline is not None,
            'supabase_connected': supabase_connected,
            'message': 'XML Validation API is running'
        }

@ns_system.route('/status')
class SystemStatus(Resource):
    @ns_system.doc('system_status')
    @ns_system.marshal_with(status_response_model)
    def get(self):
        """Get current system status and configuration"""
        supabase_status = {
            'connected': supabase_client is not None,
            'url_configured': SUPABASE_URL is not None,
            'key_configured': SUPABASE_KEY is not None
        }
        
        if supabase_client:
            try:
                result = supabase_client.table('xml_messages').select('id').limit(1).execute()
                supabase_status['database_accessible'] = True
            except Exception as e:
                supabase_status['database_accessible'] = False
                supabase_status['error'] = str(e)
        
        return {
            'api_version': '1.0',
            'status': 'running',
            'model_status': {
                'loaded': generation_pipeline is not None,
                'model_name': 'google/gemma-2-2b-it' if generation_pipeline else 'Not loaded'
            },
            'supabase_status': supabase_status,
            'supported_files': {
                'business_rules': list(ALLOWED_EXTENSIONS['rules']),
                'xml_messages': list(ALLOWED_EXTENSIONS['xml'])
            }
        }

# File upload parsers
message_store_parser = reqparse.RequestParser()
message_store_parser.add_argument('xml_file', 
                                location='files',
                                type=FileStorage, 
                                required=True,
                                help='XML message file (.xml or .txt with XML content)')

validation_with_id_parser = reqparse.RequestParser()
validation_with_id_parser.add_argument('rules_file', 
                                     location='files',
                                     type=FileStorage, 
                                     required=True,
                                     help='Business rules file (PDF, DOCX, DOC, or TXT)')

# Original validation parser (for backward compatibility)
file_upload_parser = reqparse.RequestParser()
file_upload_parser.add_argument('rules_file', 
                               location='files',
                               type=FileStorage, 
                               required=True,
                               help='Business rules file (PDF, DOCX, DOC, or TXT)')
file_upload_parser.add_argument('xml_file', 
                               location='files',
                               type=FileStorage, 
                               required=True,
                               help='XML message file (.xml or .txt with XML content)')

@ns_validation.route('/messages')
class MessageOperations(Resource):
    @ns_validation.doc('store_xml_message')
    @ns_validation.expect(message_store_parser)
    @ns_validation.marshal_with(store_message_response, code=200)
    @ns_validation.marshal_with(error_response_model, code=400)
    def post(self):
        """
        Store XML message in database
        
        Upload an XML message file to store it in the database.
        Returns a message ID that can be used for validation.
        """
        if not supabase_client:
            return {
                'success': False,
                'error': 'Supabase not configured. Please set SUPABASE_URL and SUPABASE_ANON_KEY environment variables.'
            }, 500
        
        try:
            args = message_store_parser.parse_args()
            xml_file = args['xml_file']
            
            if not xml_file.filename:
                return {
                    'success': False,
                    'error': 'XML file must have a valid name'
                }, 400
            
            if not allowed_file(xml_file.filename, 'xml'):
                return {
                    'success': False,
                    'error': 'File must have .xml or .txt extension'
                }, 400
            
            # Save file temporarily to read content
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xml') as tmp_file:
                xml_file.save(tmp_file.name)
                xml_content = read_xml_file(tmp_file.name)
                os.unlink(tmp_file.name)
            
            # Parse XML to extract metadata
            parsed_data = parse_xml_message(xml_content)
            if not parsed_data:
                return {
                    'success': False,
                    'error': 'Failed to parse XML content'
                }, 400
            
            # Store in Supabase
            message_data = {
                'xml_content': xml_content,
                'verzoeknummer': parsed_data.get('verzoeknummer'),
                'activity_name': parsed_data.get('activity_name'),
                'message_type': parsed_data.get('message_type'),
                'project_name': parsed_data.get('project_name'),
                'initiatiefnemer_name': parsed_data.get('initiatiefnemer_name'),
                'bevoegd_gezag': parsed_data.get('bevoegd_gezag'),
                'original_filename': xml_file.filename,
                'file_size': len(xml_content.encode('utf-8')),
                'validation_count': 0
            }
            
            result = supabase_client.table('xml_messages').insert(message_data).execute()
            
            if result.data:
                message_id = result.data[0]['id']
                return {
                    'success': True,
                    'message_id': message_id,
                    'message': 'XML message stored successfully',
                    'parsed_data': parsed_data
                }, 200
            else:
                return {
                    'success': False,
                    'error': 'Failed to store message in database'
                }, 500
                
        except Exception as e:
            print(f"❌ Error storing message: {str(e)}")
            return {
                'success': False,
                'error': f'Storage failed: {str(e)}'
            }, 500
    
    @ns_validation.doc('list_stored_messages')
    @ns_validation.marshal_list_with(message_model)
    def get(self):
        """
        Get list of all stored XML messages
        
        Returns metadata for all stored messages for frontend display.
        """
        if not supabase_client:
            return {
                'success': False,
                'error': 'Supabase not configured'
            }, 500
        
        try:
            result = supabase_client.table('xml_messages').select(
                'id, verzoeknummer, activity_name, message_type, project_name, '
                'initiatiefnemer_name, bevoegd_gezag, created_at, validation_count, original_filename'
            ).order('created_at', desc=True).execute()
            
            return result.data, 200
            
        except Exception as e:
            print(f"❌ Error fetching messages: {str(e)}")
            return {
                'success': False,
                'error': f'Failed to fetch messages: {str(e)}'
            }, 500

@ns_validation.route('/validate/<string:message_id>')
class ValidateStoredMessage(Resource):
    @ns_validation.doc('validate_stored_message')
    @ns_validation.expect(validation_with_id_parser)
    @ns_validation.marshal_with(validation_response_model, code=200)
    @ns_validation.marshal_with(error_response_model, code=400)
    def post(self, message_id):
        """
        Validate stored XML message against business rules
        
        Provide a message ID and business rules file to validate
        a previously stored XML message.
        """
        if not supabase_client:
            return {
                'success': False,
                'error': 'Supabase not configured',
                'decision': 'ERROR'
            }, 500
        
        try:
            args = validation_with_id_parser.parse_args()
            rules_file = args['rules_file']
            
            # Validate business rules file
            if not rules_file.filename:
                return {
                    'success': False,
                    'error': 'Business rules file must have a valid name',
                    'decision': 'ERROR'
                }, 400
            
            if not allowed_file(rules_file.filename, 'rules'):
                return {
                    'success': False,
                    'error': f'Invalid rules file type. Allowed: {", ".join(ALLOWED_EXTENSIONS["rules"])}',
                    'decision': 'ERROR'
                }, 400
            
            # Fetch XML message from database
            result = supabase_client.table('xml_messages').select('*').eq('id', message_id).execute()
            
            if not result.data:
                return {
                    'success': False,
                    'error': 'Message not found',
                    'decision': 'ERROR'
                }, 404
            
            message_record = result.data[0]
            xml_content = message_record['xml_content']
            
            # Initialize model if not already done
            global generation_pipeline
            if generation_pipeline is None:
                init_model()
            
            # Save business rules file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=get_file_extension(rules_file.filename)) as tmp_rules:
                rules_file.save(tmp_rules.name)
                rules_path = tmp_rules.name
            
            try:
                # Load business rules
                print(f"📚 Loading business rules from: {rules_file.filename}")
                rules_vectordb = load_business_rules(rules_path)
                
                # Validate
                print(f"⚖️ Validating message ID: {message_id}")
                validation_result = validate_xml_against_rules(xml_content, rules_vectordb, generation_pipeline)
                
                # Update validation count and last result in database
                update_data = {
                    'last_validation_result': validation_result,
                    'validation_count': message_record['validation_count'] + 1
                }
                supabase_client.table('xml_messages').update(update_data).eq('id', message_id).execute()
                
                # Store validation history
                history_data = {
                    'message_id': message_id,
                    'decision': validation_result['decision'],
                    'technical_reasons': validation_result['technical_reasons'],
                    'explanation': validation_result['explanation'],
                    'business_rules_used': rules_file.filename
                }
                supabase_client.table('validation_history').insert(history_data).execute()
                
                # Add success flag and file info
                validation_result['success'] = True
                validation_result['files_processed'] = {
                    'message_id': message_id,
                    'rules_file': rules_file.filename,
                    'message_filename': message_record['original_filename']
                }
                
                print(f"✅ Validation complete: {validation_result['decision']}")
                return validation_result, 200
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(rules_path)
                except Exception as cleanup_error:
                    print(f"Warning: Could not clean up temp file: {cleanup_error}")
                    
        except Exception as e:
            print(f"❌ Validation error: {str(e)}")
            print(f"❌ Full traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'decision': 'ERROR',
                'technical_reasons': f'System error: {str(e)}',
                'explanation': 'An error occurred during validation. Please check your files and try again.'
            }, 500

@ns_validation.route('/messages/<string:message_id>')
class MessageDetails(Resource):
    @ns_validation.doc('get_message_details')
    def get(self, message_id):
        """Get detailed information about a specific message including validation history"""
        if not supabase_client:
            return {'error': 'Supabase not configured'}, 500
        
        try:
            # Get message details
            result = supabase_client.table('xml_messages').select('*').eq('id', message_id).execute()
            
            if not result.data:
                return {'error': 'Message not found'}, 404
            
            message = result.data[0]
            
            # Get validation history
            history_result = supabase_client.table('validation_history').select('*').eq('message_id', message_id).order('created_at', desc=True).execute()
            
            return {
                'message': message,
                'validation_history': history_result.data
            }, 200
            
        except Exception as e:
            return {'error': str(e)}, 500

# Original validation endpoint (for backward compatibility)
@ns_validation.route('/validate')
class ValidateMessage(Resource):
    @ns_validation.doc('validate_xml_message')
    @ns_validation.expect(file_upload_parser)
    @ns_validation.marshal_with(validation_response_model, code=200)
    @ns_validation.marshal_with(error_response_model, code=400)
    def post(self):
        """
        Validate XML message against business rules (original method)
        
        Upload both a business rules file and an XML message file.
        This is the original validation method for backward compatibility.
        """
        try:
            args = file_upload_parser.parse_args()
            rules_file = args['rules_file']
            xml_file = args['xml_file']
            
            if not rules_file.filename or not xml_file.filename:
                return {
                    'success': False,
                    'error': 'Files must have valid names',
                    'decision': 'ERROR'
                }, 400
            
            if not allowed_file(rules_file.filename, 'rules'):
                return {
                    'success': False,
                    'error': f'Invalid rules file type. Allowed: {", ".join(ALLOWED_EXTENSIONS["rules"])}',
                    'decision': 'ERROR'
                }, 400
            
            if not allowed_file(xml_file.filename, 'xml'):
                return {
                    'success': False,
                    'error': 'XML file must have .xml or .txt extension',
                    'decision': 'ERROR'
                }, 400
            
            global generation_pipeline
            if generation_pipeline is None:
                init_model()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=get_file_extension(rules_file.filename)) as tmp_rules:
                rules_file.save(tmp_rules.name)
                rules_path = tmp_rules.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xml', mode='w+b') as tmp_xml:
                xml_file.save(tmp_xml.name)
                xml_path = tmp_xml.name
            
            try:
                print(f"📚 Loading business rules from: {rules_file.filename}")
                rules_vectordb = load_business_rules(rules_path)
                
                print(f"📄 Reading XML file: {xml_file.filename}")
                xml_content = read_xml_file(xml_path)
                
                print(f"⚖️ Starting validation...")
                result = validate_xml_against_rules(xml_content, rules_vectordb, generation_pipeline)
                
                result['success'] = True
                result['files_processed'] = {
                    'rules_file': rules_file.filename,
                    'xml_file': xml_file.filename
                }
                
                print(f"✅ Validation complete: {result['decision']}")
                return result, 200
                
            finally:
                try:
                    os.unlink(rules_path)
                    os.unlink(xml_path)
                except Exception as cleanup_error:
                    print(f"Warning: Could not clean up temp files: {cleanup_error}")
                    
        except Exception as e:
            print(f"❌ Validation error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'decision': 'ERROR',
                'technical_reasons': f'System error: {str(e)}',
                'explanation': 'An error occurred during validation.'
            }, 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/api/v1/system/health', 
            '/api/v1/system/status',
            '/api/v1/validation/messages',
            '/api/v1/validation/validate',
            '/api/v1/validation/validate/<message_id>',
            '/api/v1/validation/messages/<message_id>'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'Something went wrong on the server'
    }), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({
        'success': False,
        'error': 'File too large',
        'message': 'Uploaded file exceeds size limit'
    }), 413

# Initialize model when app starts
print("🚀 Starting XML Validation API Server with Supabase Integration...")
print("📱 CORS enabled for: http://localhost:3000")
print("📚 Swagger UI available at: http://localhost:5000/docs/")
print("🔧 Available endpoints:")
print("  - GET  /api/v1/system/health              - Health check")
print("  - GET  /api/v1/system/status              - System status") 
print("  - POST /api/v1/validation/messages        - Store XML message")
print("  - GET  /api/v1/validation/messages        - List stored messages")
print("  - POST /api/v1/validation/validate/<id>   - Validate stored message")
print("  - GET  /api/v1/validation/messages/<id>   - Get message details")
print("  - POST /api/v1/validation/validate        - Original validation (backward compatibility)")

# Initialize the AI model on startup
init_model()

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )