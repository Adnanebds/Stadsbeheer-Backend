# Complete Backend API with CORS for Frontend Integration
from flask import Flask, request, jsonify
from flask_cors import CORS
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

# Global variable to store the AI model (loaded once)
generation_pipeline = None

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'rules': {'pdf', 'docx', 'doc', 'txt'},
    'xml': {'xml'}
}

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
            loader = PyPDFLoader(file_path)
        elif file_extension in ['docx', 'doc']:
            loader = Docx2txtLoader(file_path)
        else:
            # Try different encodings for text files
            try:
                loader = TextLoader(file_path, encoding='utf-8')
            except:
                try:
                    loader = TextLoader(file_path, encoding='latin-1')
                except:
                    # Fallback to unstructured loader which is more robust
                    loader = UnstructuredFileLoader(file_path)
        
        documents = loader.load()
        print(f"Successfully loaded {len(documents)} document(s) from {file_path}")
        
        # Split documents with focus on activity sections
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Larger chunks to capture complete rule sections
            chunk_overlap=300,  # Generous overlap to avoid splitting rules
            separators=["**Activiteit:", "\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        
        # Print a sample chunk to verify content
        if chunks:
            print("\nSample chunk content:")
            print(chunks[0].page_content[:200] + "..." if len(chunks[0].page_content) > 200 else chunks[0].page_content)
        
        # Initialize vector store
        print("Creating embeddings... (this may take a moment)")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(chunks, embeddings)
        print("✅ Vector database created successfully")
        
        return vectordb
    
    except Exception as e:
        print(f"Error loading document: {e}")
        # A more robust fallback approach
        print("Trying alternative loading method...")
        try:
            # Read raw bytes and decode with a very permissive encoding
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Try to decode with different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    text = content.decode(encoding, errors='replace')
                    print(f"Successfully decoded with {encoding}")
                    
                    # Create a document
                    from langchain.schema import Document
                    document = Document(page_content=text, metadata={"source": file_path})
                    
                    # Split into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500,
                        chunk_overlap=300
                    )
                    chunks = text_splitter.split_documents([document])
                    
                    # Initialize vector store
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    vectordb = FAISS.from_documents(chunks, embeddings)
                    
                    return vectordb
                except UnicodeDecodeError:
                    continue
            
            raise RuntimeError("Could not decode the file with any encoding")
            
        except Exception as inner_e:
            print(f"Alternative loading also failed: {inner_e}")
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
        print(text[:200] + "..." if len(text) > 200 else text)
        
        # If this is a section with informatieplicht and we're looking for informatie requirements
        if "informatieplicht" in text.lower() and "informatie" in message_type_lower:
            print("Found informatieplicht section relevant to Informatie message type")
            
            # Extract field requirements using regex
            field_pattern = r"(?:gegevens en bescheiden verstrekt over|Een .+ bevat):(.*?)(?=\n\d|\Z)"
            field_matches = re.findall(field_pattern, text, re.DOTALL)
            
            for match in field_matches:
                # Extract individual fields from lettered list
                field_items = re.findall(r"([a-z]\.\s*)(.*?)(?=\n[a-z]\.|$)", match, re.DOTALL)
                for _, field in field_items:
                    clean_field = field.strip()
                    print(f"  ✅ Found required field: {clean_field}")
                    requirements['required_fields'].append(clean_field)
            
            # Check for attachment requirements
            if "evaluatieverslag" in text.lower():
                print("  📎 Found attachment requirement: evaluatieverslag")
                requirements['required_attachments'].append("evaluatieverslag")
        
        # If this is a melding section and we're looking for melding requirements
        elif "melding" in text.lower() and "melding" in message_type_lower:
            print("Found melding section relevant to Melding message type")
            
            # Extract field requirements for meldingen
            field_pattern = r"(?:Een melding bevat:)(.*?)(?=\n\d|\Z)"
            field_matches = re.findall(field_pattern, text, re.DOTALL)
            
            for match in field_matches:
                # Extract individual fields from lettered list
                field_items = re.findall(r"([a-z]\.\s*)(.*?)(?=\n[a-z]\.|$)", match, re.DOTALL)
                for _, field in field_items:
                    clean_field = field.strip()
                    print(f"  ✅ Found required field: {clean_field}")
                    requirements['required_fields'].append(clean_field)
            
            # Check for attachment requirements in melding
            if "evaluatieverslag" in text.lower():
                print("  📎 Found attachment requirement: evaluatieverslag")
                requirements['required_attachments'].append("evaluatieverslag")
    
    return requirements

def validate_message(message_data, requirements):
    print("\n⚖️ Validating message against requirements:")
    
    # Check if all required fields are present
    missing_fields = []
    for field in requirements['required_fields']:
        field_found = False
        
        # Special case for boundary location information
        if "begrenzing van de locatie" in field.lower() and message_data.get('has_coordinates', False):
            field_found = True
            print(f"  ✅ Required field found: '{field}' (via coordinates in message)")
            continue
        
        # Check specifications
        for spec in message_data['specifications']:
            if (spec['answer'] and field.lower() in spec['answer'].lower()) or (
                spec['question'] and field.lower() in spec['question'].lower()
            ):
                field_found = True
                print(f"  ✅ Required field found: '{field}'")
                break
        
        if not field_found:
            print(f"  ❌ Required field missing: '{field}'")
            missing_fields.append(field)
    
    # Check if all required attachments are present
    missing_attachments = []
    for attachment in requirements['required_attachments']:
        if not message_data['attachments']:
            print(f"  ❌ Required attachment missing: '{attachment}' (no attachments present)")
            missing_attachments.append(attachment)
            continue
        
        attachment_found = False
        for att in message_data['attachments']:
            if attachment.lower() in att.lower():
                attachment_found = True
                print(f"  ✅ Required attachment found: '{attachment}' in '{att}'")
                break
        
        if not attachment_found:
            print(f"  ❌ Required attachment missing: '{attachment}'")
            missing_attachments.append(attachment)
    
    # Determine if message is valid
    is_valid = len(missing_fields) == 0 and len(missing_attachments) == 0
    
    return {
        'is_valid': is_valid,
        'missing_fields': missing_fields,
        'missing_attachments': missing_attachments
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
    
    # 5. Generate explanation with Gemma 2-2b
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

# API Routes

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': generation_pipeline is not None,
        'message': 'XML Validation API is running'
    })

@app.route('/validate', methods=['POST'])
def validate_message_endpoint():
    """
    Validate XML message against business rules
    Expects multipart/form-data with 'rules_file' and 'xml_file'
    """
    try:
        # Check if files are present
        if 'rules_file' not in request.files or 'xml_file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Both rules_file and xml_file are required',
                'decision': 'ERROR',
                'technical_reasons': 'Missing files',
                'explanation': 'Please upload both business rules and XML message files'
            }), 400
        
        rules_file = request.files['rules_file']
        xml_file = request.files['xml_file']
        
        # Check if files have valid names
        if rules_file.filename == '' or xml_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Files must have valid names',
                'decision': 'ERROR',
                'technical_reasons': 'Empty filenames',
                'explanation': 'Please upload files with valid names'
            }), 400
        
        # Check file extensions
        if not allowed_file(rules_file.filename, 'rules'):
            return jsonify({
                'success': False,
                'error': f'Invalid rules file type. Allowed: {", ".join(ALLOWED_EXTENSIONS["rules"])}',
                'decision': 'ERROR',
                'technical_reasons': 'Invalid file type',
                'explanation': 'Business rules file must be PDF, DOCX, DOC, or TXT'
            }), 400
        
        if not allowed_file(xml_file.filename, 'xml'):
            return jsonify({
                'success': False,
                'error': 'XML file must have .xml extension',
                'decision': 'ERROR',
                'technical_reasons': 'Invalid file type',
                'explanation': 'Message file must be XML format'
            }), 400
        
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
            return jsonify(result), 200
            
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
        return jsonify({
            'success': False,
            'error': str(e),
            'decision': 'ERROR', 
            'technical_reasons': f'System error: {str(e)}',
            'explanation': 'An error occurred during validation. Please check your files and try again.'
        }), 500

@app.route('/upload', methods=['POST'])
def upload_files():
    """
    Alternative upload endpoint that just processes and returns file info
    Can be used for file validation before sending to /validate
    """
    try:
        uploaded_files = {}
        
        # Process rules file if present
        if 'rules_file' in request.files:
            rules_file = request.files['rules_file']
            if rules_file.filename != '':
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
        if 'xml_file' in request.files:
            xml_file = request.files['xml_file']
            if xml_file.filename != '':
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
                        'error': 'File must have .xml extension'
                    }
        
        return jsonify({
            'success': True,
            'message': 'Files processed successfully',
            'files': uploaded_files
        }), 200
        
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Upload failed: {str(e)}'
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get current system status"""
    return jsonify({
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
    })

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': ['/health', '/validate', '/upload', '/status']
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
print("🚀 Starting XML Validation API Server...")
print("📱 CORS enabled for: http://localhost:3000")
print("🔧 Available endpoints:")
print("  - GET  /health   - Health check")
print("  - POST /validate - Validate XML message")
print("  - POST /upload   - Upload and validate files")
print("  - GET  /status   - System status")

# Initialize the AI model on startup
init_model()

if __name__ == '__main__':
    # Run the Flask development server
    app.run(
        host='0.0.0.0',  # Allow connections from any IP
        port=5000,       # Default Flask port
        debug=True       # Enable debug mode for development
    )