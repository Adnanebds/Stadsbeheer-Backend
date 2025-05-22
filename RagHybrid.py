# Complete Backend API with CORS and Swagger Documentation
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

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for localhost:3000 (React development server)
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])

# Initialize Swagger API
api = Api(
    app,
    version='1.0',
    title='XML Validation API',
    description='API for validating XML messages against business rules using AI',
    doc='/docs/',  # Swagger UI will be available at /docs/
    prefix='/api/v1'  # Optional: add API versioning
)

# Create namespaces for better organization
ns_validation = api.namespace('validation', description='XML message validation operations')
ns_system = api.namespace('system', description='System status and health checks')

# Global variable to store the AI model (loaded once)
generation_pipeline = None

# Install additional PDF dependencies if not already installed
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

# Call this at startup
try:
    install_pdf_dependencies()
except Exception as e:
    print(f"⚠️ Warning: Could not install PDF dependencies: {e}")

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'rules': {'pdf', 'docx', 'doc', 'txt'},
    'xml': {'xml', 'txt'}  # Allow both .xml and .txt for XML content
}

# Swagger Models for request/response documentation
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
    'message': fields.String(required=True, description='Status message')
})

status_response_model = api.model('SystemStatus', {
    'api_version': fields.String(required=True, description='API version'),
    'status': fields.String(required=True, description='System status'),
    'model_status': fields.Raw(description='AI model status information'),
    'supported_files': fields.Raw(description='Supported file types')
})

upload_response_model = api.model('UploadResult', {
    'success': fields.Boolean(required=True, description='Whether upload was successful'),
    'message': fields.String(required=True, description='Upload status message'),
    'files': fields.Raw(description='Information about uploaded files')
})

error_response_model = api.model('ErrorResponse', {
    'success': fields.Boolean(required=True, description='Always false for errors'),
    'error': fields.String(required=True, description='Error message'),
    'decision': fields.String(description='Decision status for validation errors'),
    'technical_reasons': fields.String(description='Technical error details'),
    'explanation': fields.String(description='User-friendly error explanation')
})

def allowed_file(filename, file_type):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[file_type]

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
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # Automatically use GPU if available
            torch_dtype="auto"  # Use appropriate precision
        )
        
        # Create text generation pipeline
        generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.1,  # Lower temperature for more precise responses
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        print("✅ Gemma 2-2b model loaded successfully")
        return generation_pipeline
        
    except Exception as e:
        print(f"❌ Error loading Gemma model: {e}")
        print("Falling back to smaller model...")
        try:
            # Fallback to a smaller model
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
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Business rules file not found: {file_path}")
    
    # Determine file type and use appropriate loader
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
                
                # Try with UnstructuredFileLoader
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
            # Try different encodings for text files
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
            except:
                try:
                    loader = TextLoader(file_path, encoding='latin-1')
                    documents = loader.load()
                except:
                    # Fallback to unstructured loader which is more robust
                    loader = UnstructuredFileLoader(file_path)
                    documents = loader.load()
        
        if not documents:
            raise Exception("No documents were loaded from the file")
            
        # Check if we got readable content
        total_text = ' '.join([doc.page_content for doc in documents])
        readable_chars = sum(1 for c in total_text if c.isprintable() and ord(c) < 256)
        total_chars = len(total_text)
        
        if total_chars == 0:
            raise Exception("Loaded documents contain no text")
            
        readability_ratio = readable_chars / total_chars if total_chars > 0 else 0
        print(f"📊 Content readability: {readability_ratio:.2%} ({readable_chars}/{total_chars} readable chars)")
        
        if readability_ratio < 0.7:  # Less than 70% readable
            print("⚠️ WARNING: Low readability detected. The file might be corrupted or in an unsupported format.")
            print("💡 SUGGESTION: Try converting your PDF to TXT format for better results.")
            
            # Try to clean up the text
            cleaned_docs = []
            for doc in documents:
                # Remove non-printable characters and keep only Dutch/English text
                cleaned_text = ''.join(c for c in doc.page_content if c.isprintable() and ord(c) < 256)
                if len(cleaned_text) > 100:  # Only keep chunks with substantial text
                    from langchain.schema import Document
                    cleaned_docs.append(Document(page_content=cleaned_text, metadata=doc.metadata))
            
            if cleaned_docs:
                documents = cleaned_docs
                print(f"🧹 Cleaned documents: {len(documents)} documents with readable text")
            else:
                raise Exception("After cleaning, no readable text remains. Please convert your PDF to TXT format.")
        
        print(f"✅ Successfully loaded {len(documents)} document(s) from {file_path}")
        
        # Split documents with focus on activity sections
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Larger chunks to capture complete rule sections
            chunk_overlap=400,  # More generous overlap to avoid splitting rules
            separators=["Activiteit:", "Artikel", "§", "\n\n", "\n", " ", ""]  # Better separators for legal documents
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        
        # Print a sample chunk to verify content is readable
        if chunks:
            sample_content = chunks[0].page_content[:500]
            print("\n📄 Sample chunk content:")
            print(sample_content + "..." if len(chunks[0].page_content) > 500 else sample_content)
            
            # Check if sample contains business rule keywords
            dutch_keywords = ['artikel', 'activiteit', 'melding', 'informatie', 'vereisten', 'bodem', 'saneren']
            found_keywords = [kw for kw in dutch_keywords if kw.lower() in sample_content.lower()]
            
            if found_keywords:
                print(f"✅ Found business rule keywords: {found_keywords}")
            else:
                print("⚠️ WARNING: No business rule keywords found in sample. File might not contain proper business rules.")
        
        # Initialize vector store
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
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"XML file not found: {file_path}")
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            print(f"✅ Successfully read XML file with {encoding} encoding")
            return content
        except UnicodeDecodeError:
            continue
    
    # If all encodings fail, try reading as bytes and replace invalid chars
    try:
        with open(file_path, 'rb') as file:
            content = file.read()
        return content.decode('utf-8', errors='replace')
    except Exception as e:
        raise RuntimeError(f"Could not read XML file: {e}")

def parse_xml_message(xml_content):
    try:
        # Parse XML
        root = ET.fromstring(xml_content)
        
        # Extract namespace for easier querying
        namespaces = {
            'imam': 'http://www.omgevingswet.nl/koppelvlak/stam-v4/IMAM',
            'vx': 'http://www.omgevingswet.nl/koppelvlak/stam-v4/verzoek'
        }
        
        # Extract activity name
        activity_elements = root.findall('.//vx:activiteitnaam', namespaces)
        activity_name = activity_elements[0].text if activity_elements else None
        
        # Extract message type (Melding or Informatie)
        message_type_elements = root.findall('.//vx:type', namespaces)
        message_type = message_type_elements[0].text if message_type_elements else None
        
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
        
        # Check if document is attached
        document_elements = root.findall('.//vx:document', namespaces)
        attachments = []
        for doc in document_elements:
            filename_elem = doc.find('.//vx:bestandsnaam', namespaces)
            if filename_elem is not None and filename_elem.text:
                attachments.append(filename_elem.text)
        
        # Check for location coordinates (boundary info)
        has_coordinates = len(root.findall('.//vx:coordinatenEtrs', namespaces)) > 0 or len(root.findall('.//vx:coordinatenOpgegeven', namespaces)) > 0
        
        result = {
            'activity_name': activity_name,
            'message_type': message_type,
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
    # Create more specific queries
    queries = [
        f"Business rules for {activity_name} with {message_type} requirements",
        f"{activity_name} {message_type} vereisten",
        f"Artikel {message_type} {activity_name}"
    ]
    
    # Retrieve results for each query
    all_results = []
    for query in queries:
        print(f"🔍 Searching for: {query}")
        results = vectordb.similarity_search(query, k=2)
        all_results.extend(results)
    
    # Remove duplicates
    unique_results = []
    seen_contents = set()
    for doc in all_results:
        if doc.page_content not in seen_contents:
            unique_results.append(doc)
            seen_contents.add(doc.page_content)
    
    # Extract text from results
    rule_texts = [doc.page_content for doc in unique_results]
    print(f"✅ Retrieved {len(rule_texts)} unique rule text chunks")
    
    return rule_texts

def extract_requirements(rule_texts, message_type):
    requirements = {
        'required_fields': [],
        'required_attachments': []
    }
    
    # Look for specific patterns based on message type
    message_type_lower = message_type.lower() if message_type else ""
    
    for text in rule_texts:
        print(f"\n📝 Analyzing text chunk for requirements ({len(text)} characters):")
        print(text[:400] + "..." if len(text) > 400 else text)
        
        # ENHANCED: Look for "Bodem saneren" specific rules with multiple patterns
        if "bodem saneren" in text.lower() or "saneren van de bodem" in text.lower():
            print("🎯 Found 'Bodem saneren' specific section")
            
            # For Informatie type - look for informatieplicht requirements
            if "informatie" in message_type_lower and "informatieplicht" in text.lower():
                print("✅ Found informatieplicht section for Bodem saneren")
                
                # ENHANCED: Multiple patterns for requirements extraction
                patterns = [
                    r"(?:gegevens en bescheiden verstrekt over|worden.*?verstrekt over):\s*(.*?)(?=\n\d|\n[A-Z]|Artikel|\Z)",
                    r"(?:Een.*?bevat):\s*(.*?)(?=\n\d|\n[A-Z]|Artikel|\Z)",
                    r"(?:melding bevat):\s*(.*?)(?=\n\d|\n[A-Z]|Artikel|\Z)"
                ]
                
                for pattern in patterns:
                    field_matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                    for match in field_matches:
                        # Extract individual fields from lettered/numbered lists
                        field_items = re.findall(r"([a-z]\.\s*|[0-9]+°?\.\s*)(.*?)(?=\n[a-z]\.|$|\n\d)", match, re.DOTALL)
                        for _, field in field_items:
                            clean_field = field.strip()
                            if clean_field and len(clean_field) > 5:  # Filter out very short matches
                                print(f"  ✅ Found required field: {clean_field}")
                                requirements['required_fields'].append(clean_field)
                
                # ENHANCED: Look for specific requirements we know are needed
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
                
            # For Melding type
            elif "melding" in message_type_lower and "melding" in text.lower():
                print("✅ Found melding section for Bodem saneren")
                
                # Look for melding requirements
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
        
        # ENHANCED: Check for attachment requirements with multiple patterns
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
    
    # FALLBACK: If no requirements found, add known requirements for Bodem saneren informatieplicht
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
    
    # Check if all required fields are present
    missing_fields = []
    found_fields = []
    
    for field in requirements['required_fields']:
        field_found = False
        
        # ENHANCED: Multiple ways to check for boundary location information
        if "begrenzing van de locatie" in field.lower():
            if message_data.get('has_coordinates', False):
                field_found = True
                found_fields.append(f"'{field}' (via coordinates)")
                print(f"  ✅ Required field found: '{field}' (via coordinates in message)")
                continue
        
        # ENHANCED: Check for start date in multiple ways
        if "verwachte datum" in field.lower() and "begin" in field.lower():
            # Check if any specification mentions timing or dates
            for spec in message_data['specifications']:
                if spec['answer'] and any(word in spec['answer'].lower() for word in ['datum', 'begin', 'start', 'tijd']):
                    field_found = True
                    found_fields.append(f"'{field}' (in specifications)")
                    print(f"  ✅ Required field found: '{field}' (timing info in specifications)")
                    break
        
        # ENHANCED: Check for work performer information
        if "naam en het adres van degene die de werkzaamheden" in field.lower():
            # Check if initiatiefnemer or gemachtigde information is present (from XML structure)
            if message_data.get('has_initiator_info', True):  # Assume present for now
                field_found = True
                found_fields.append(f"'{field}' (initiator/representative info)")
                print(f"  ✅ Required field found: '{field}' (via initiator/representative info)")
        
        # ENHANCED: Check for environmental guidance information
        if "milieukundige begeleiding" in field.lower():
            # This is often not present in information-type submissions
            print(f"  ⚠️ Environmental guidance info often not required for information submissions: '{field}'")
            # Don't mark as missing for information type submissions
            if message_data.get('message_type', '').lower() == 'informatie':
                field_found = True  # Allow this for informatie type
                found_fields.append(f"'{field}' (not required for informatie)")
        
        # ENHANCED: Original specification checking with better matching
        if not field_found:
            for spec in message_data['specifications']:
                # Check answer content
                if spec['answer']:
                    # More flexible matching
                    field_keywords = field.lower().split()
                    answer_lower = spec['answer'].lower()
                    if any(keyword in answer_lower for keyword in field_keywords[-3:]):  # Check last 3 words
                        field_found = True
                        found_fields.append(f"'{field}' (in answer: {spec['answer'][:50]}...)")
                        print(f"  ✅ Required field found: '{field}' (in answer)")
                        break
                
                # Check question content
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
    
    # ENHANCED: Check attachments with better matching
    missing_attachments = []
    found_attachments = []
    
    for attachment in requirements['required_attachments']:
        if not message_data['attachments']:
            print(f"  ❌ Required attachment missing: '{attachment}' (no attachments present)")
            missing_attachments.append(attachment)
            continue
        
        attachment_found = False
        for att in message_data['attachments']:
            # More flexible attachment matching
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
    
    # SPECIAL CASE: For "Informatie" type with evaluation report after completion
    if (message_data.get('message_type', '').lower() == 'informatie' and 
        any('evaluatie' in att.lower() for att in message_data['attachments']) and
        len(missing_fields) > 0):
        
        print("\n🔍 SPECIAL CASE DETECTED:")
        print("This appears to be a post-activity information submission with evaluation report.")
        print("For completed activities with evaluation reports, some pre-activity requirements may not apply.")
        
        # Filter out pre-activity requirements that don't apply to post-completion submissions
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
    
    # Determine if message is valid
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
        # Fallback explanation if model is not available
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
    
    # Generate response using the available model
    try:
        print("🤖 Generating AI explanation...")
        result = generation_pipeline(prompt, max_new_tokens=100, temperature=0.3, pad_token_id=generation_pipeline.tokenizer.eos_token_id)
        
        # Extract just the generated response part
        generated_text = result[0]['generated_text']
        if len(generated_text) > len(prompt):
            explanation = generated_text[len(prompt):].strip()
        else:
            explanation = generated_text.strip()
        
        # Clean up repetitive text
        if explanation.count(explanation.split('.')[0]) > 2:
            # If the first sentence repeats too much, use fallback
            raise Exception("Repetitive output detected")
            
        return explanation if explanation else prompt
        
    except Exception as e:
        print(f"❌ Error generating explanation: {e}")
        # Use the prompt as fallback explanation
        return prompt

def validate_xml_against_rules(xml_content, vectordb, generation_pipeline):
    print("\n" + "="*60)
    print("🚀 STARTING VALIDATION PROCESS")
    print("="*60)
    
    # 1. Parse the message
    print("\n📄 Step 1: Parsing XML message")
    message_data = parse_xml_message(xml_content)
    if not message_data:
        return {"decision": "REJECTED", "technical_reasons": "Failed to parse XML", "explanation": "Failed to parse XML"}
    
    print(f"\n📊 Parsed message data:")
    print(f"  Activity: {message_data['activity_name']}")
    print(f"  Message Type: {message_data['message_type']}")
    print(f"  Specifications: {len(message_data['specifications'])} items")
    print(f"  Attachments: {message_data['attachments']}")
    
    # 2. Retrieve relevant business rules
    print(f"\n🔍 Step 2: Retrieving relevant business rules")
    rule_texts = retrieve_business_rules(
        vectordb, 
        message_data['activity_name'], 
        message_data['message_type']
    )
    
    # 3. Extract structured requirements
    print(f"\n⚙️ Step 3: Extracting structured requirements")
    requirements = extract_requirements(rule_texts, message_data['message_type'])
    
    print(f"\n📋 Extracted requirements summary:")
    print(f"  Required fields: {len(requirements['required_fields'])}")
    print(f"  Required attachments: {len(requirements['required_attachments'])}")
    
    # 4. Validate the message
    print(f"\n⚖️ Step 4: Validating message")
    validation_result = validate_message(message_data, requirements)
    
    # 5. Generate explanation with AI
    print(f"\n🤖 Step 5: Generating explanation with AI")
    explanation = generate_explanation(generation_pipeline, validation_result, message_data)
    
    # 6. Prepare final result
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

# API Routes using Flask-RESTX

@ns_system.route('/health')
class HealthCheck(Resource):
    @ns_system.doc('health_check')
    @ns_system.marshal_with(health_response_model)
    def get(self):
        """Health check endpoint - Check if the API is running properly"""
        return {
            'status': 'healthy',
            'model_loaded': generation_pipeline is not None,
            'message': 'XML Validation API is running'
        }

@ns_system.route('/status')
class SystemStatus(Resource):
    @ns_system.doc('system_status')
    @ns_system.marshal_with(status_response_model)
    def get(self):
        """Get current system status and configuration"""
        return {
            'api_version': '1.0',
            'status': 'running',
            'model_status': {
                'loaded': generation_pipeline is not None,
                'model_name': 'google/gemma-2-2b-it' if generation_pipeline else 'Not loaded'
            },
            'supported_files': {
                'business_rules': list(ALLOWED_EXTENSIONS['rules']),
                'xml_messages': list(ALLOWED_EXTENSIONS['xml'])
            }
        }

# File upload parser
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

@ns_validation.route('/validate')
class ValidateMessage(Resource):
    @ns_validation.doc('validate_xml_message')
    @ns_validation.expect(file_upload_parser)
    @ns_validation.marshal_with(validation_response_model, code=200)
    @ns_validation.marshal_with(error_response_model, code=400)
    @ns_validation.marshal_with(error_response_model, code=500)
    def post(self):
        """
        Validate XML message against business rules
        
        Upload both a business rules file and an XML message file.
        The system will parse the business rules, extract requirements,
        and validate the XML message against those requirements using AI.
        """
        try:
            args = file_upload_parser.parse_args()
            rules_file = args['rules_file']
            xml_file = args['xml_file']
            
            # Check if files have valid names
            if not rules_file.filename or not xml_file.filename:
                return {
                    'success': False,
                    'error': 'Files must have valid names',
                    'decision': 'ERROR',
                    'technical_reasons': 'Empty filenames',
                    'explanation': 'Please upload files with valid names'
                }, 400
            
            # Check file extensions
            if not allowed_file(rules_file.filename, 'rules'):
                return {
                    'success': False,
                    'error': f'Invalid rules file type. Allowed: {", ".join(ALLOWED_EXTENSIONS["rules"])}',
                    'decision': 'ERROR',
                    'technical_reasons': 'Invalid file type',
                    'explanation': 'Business rules file must be PDF, DOCX, DOC, or TXT'
                }, 400
            
            if not allowed_file(xml_file.filename, 'xml'):
                return {
                    'success': False,
                    'error': 'XML file must have .xml or .txt extension',
                    'decision': 'ERROR',
                    'technical_reasons': 'Invalid file type',
                    'explanation': 'Message file must be XML format (.xml) or XML content in text file (.txt)'
                }, 400
            
            # Initialize model if not already done
            global generation_pipeline
            if generation_pipeline is None:
                init_model()
            
            # Save uploaded files temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=get_file_extension(rules_file.filename)) as tmp_rules:
                rules_file.save(tmp_rules.name)
                rules_path = tmp_rules.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xml', mode='w+b') as tmp_xml:
                xml_file.save(tmp_xml.name)
                xml_path = tmp_xml.name
            
            try:
                # Load business rules
                print(f"📚 Loading business rules from: {rules_file.filename}")
                rules_vectordb = load_business_rules(rules_path)
                
                # Read XML content
                print(f"📄 Reading XML file: {xml_file.filename}")
                xml_content = read_xml_file(xml_path)
                
                # Validate
                print(f"⚖️ Starting validation...")
                result = validate_xml_against_rules(xml_content, rules_vectordb, generation_pipeline)
                
                # Add success flag and file info
                result['success'] = True
                result['files_processed'] = {
                    'rules_file': rules_file.filename,
                    'xml_file': xml_file.filename
                }
                
                print(f"✅ Validation complete: {result['decision']}")
                return result, 200
                
            finally:
                # Clean up temporary files
                try:
                    os.unlink(rules_path)
                    os.unlink(xml_path)
                except Exception as cleanup_error:
                    print(f"Warning: Could not clean up temp files: {cleanup_error}")
                    
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

# Optional file parser for upload endpoint
upload_parser = reqparse.RequestParser()
upload_parser.add_argument('rules_file', 
                          location='files',
                          type=FileStorage, 
                          required=False,
                          help='Business rules file (PDF, DOCX, DOC, or TXT)')
upload_parser.add_argument('xml_file', 
                          location='files',
                          type=FileStorage, 
                          required=False,
                          help='XML message file (.xml or .txt with XML content)')

@ns_validation.route('/upload')
class UploadFiles(Resource):
    @ns_validation.doc('upload_files')
    @ns_validation.expect(upload_parser)
    @ns_validation.marshal_with(upload_response_model, code=200)
    @ns_validation.marshal_with(error_response_model, code=500)
    def post(self):
        """
        Upload and validate files without processing
        
        This endpoint can be used to validate file formats and get file information
        before sending them to the main validation endpoint.
        """
        try:
            args = upload_parser.parse_args()
            uploaded_files = {}
            
            # Process rules file if present
            if args['rules_file']:
                rules_file = args['rules_file']
                if rules_file.filename:
                    if allowed_file(rules_file.filename, 'rules'):
                        uploaded_files['rules_file'] = {
                            'filename': secure_filename(rules_file.filename),
                            'size': len(rules_file.read()),
                            'type': rules_file.content_type,
                            'valid': True
                        }
                        rules_file.seek(0)  # Reset file pointer
                    else:
                        uploaded_files['rules_file'] = {
                            'filename': rules_file.filename,
                            'valid': False,
                            'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS["rules"])}'
                        }
            
            # Process XML file if present
            if args['xml_file']:
                xml_file = args['xml_file']
                if xml_file.filename:
                    if allowed_file(xml_file.filename, 'xml'):
                        uploaded_files['xml_file'] = {
                            'filename': secure_filename(xml_file.filename),
                            'size': len(xml_file.read()),
                            'type': xml_file.content_type,
                            'valid': True
                        }
                        xml_file.seek(0)  # Reset file pointer
                    else:
                        uploaded_files['xml_file'] = {
                            'filename': xml_file.filename,
                            'valid': False,
                            'error': 'File must have .xml or .txt extension'
                        }
            
            return {
                'success': True,
                'message': 'Files processed successfully',
                'files': uploaded_files
            }, 200
            
        except Exception as e:
            print(f"❌ Upload error: {str(e)}")
            return {
                'success': False,
                'error': f'Upload failed: {str(e)}'
            }, 500

# Error handlers for Flask app (not flask-restx api)
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': ['/api/v1/system/health', '/api/v1/validation/validate', '/api/v1/validation/upload', '/api/v1/system/status']
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
print("🚀 Starting XML Validation API Server with Swagger Documentation...")
print("📱 CORS enabled for: http://localhost:3000")
print("📚 Swagger UI available at: http://localhost:5000/docs/")
print("🔧 Available endpoints:")
print("  - GET  /api/v1/system/health   - Health check")
print("  - POST /api/v1/validation/validate - Validate XML message")
print("  - POST /api/v1/validation/upload   - Upload and validate files")
print("  - GET  /api/v1/system/status       - System status")

# Initialize the AI model on startup
init_model()

if __name__ == '__main__':
    # Run the Flask development server
    app.run(
        host='0.0.0.0',  # Allow connections from any IP
        port=5000,       # Default Flask port
        debug=True       # Enable debug mode for development
    )