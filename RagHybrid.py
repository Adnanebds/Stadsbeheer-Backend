# Complete Backend API with Rule Management - Using Supabase Storage
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restx import Api, Resource, fields, reqparse
from werkzeug.datastructures import FileStorage
import os
import tempfile
import json
import traceback
import re
import xml.etree.ElementTree as ET
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredFileLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
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
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("WARNING: Supabase credentials not found in environment variables")
    print("Please set SUPABASE_URL and SUPABASE_KEY")
    supabase_client = None
else:
    try:
        supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase client initialized successfully")
    except Exception as e:
        print(f"Failed to initialize Supabase client: {e}")
        supabase_client = None

# Initialize Swagger API
api = Api(
    app,
    version='1.6',
    title='XML Validation API with Supabase Storage',
    description='Lightweight API for validating XML messages against business rules using Supabase Storage for file management.',
    doc='/docs/',
    prefix='/api/v1'
)

# Create namespaces
ns_validation = api.namespace('validation', description='XML message validation operations')
ns_rules = api.namespace('rules', description='Business rules management operations')
ns_system = api.namespace('system', description='System status and health checks')

# Storage bucket name
STORAGE_BUCKET = 'business-rules'

# Install PDF dependencies
import subprocess
import sys

def install_pdf_dependencies():
    """Install required PDF processing libraries"""
    try:
        import pypdf
    except ImportError:
        print("Installing pypdf...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])

    try:
        import pdfplumber
    except ImportError:
        print("Installing pdfplumber...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber"])

try:
    install_pdf_dependencies()
except Exception as e:
    print(f"Warning: Could not install PDF dependencies: {e}")

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'rules': {'pdf', 'docx', 'doc', 'txt'},
    'xml': {'xml', 'txt'}
}

# --- Swagger Models ---
validation_detail_item_model = api.model('ValidationDetailItem', {
    'requirement': fields.String(), 'status': fields.String(enum=['MET', 'MISSING']),
    'reason': fields.String(), 'source_quote': fields.String()
})
validation_details_model = api.model('ValidationDetails', {
    'checked_requirements': fields.List(fields.Nested(validation_detail_item_model)),
    'checked_attachments': fields.List(fields.Nested(validation_detail_item_model))
})
lightweight_validation_response_model = api.model('LightweightValidationResult', {
    'success': fields.Boolean(required=True), 'decision': fields.String(required=True, enum=['ACCEPTED', 'REJECTED', 'ERROR']),
    'summary_explanation': fields.String(required=True), 'validation_details': fields.Nested(validation_details_model),
    'rule_used': fields.Raw(description='Information about the business rule used')
})
health_response_model = api.model('HealthCheck', {
    'status': fields.String(required=True), 'supabase_connected': fields.Boolean(required=True),
    'message': fields.String(required=True)
})
status_response_model = api.model('SystemStatus', {
    'api_version': fields.String(required=True), 'status': fields.String(required=True),
    'supabase_status': fields.Raw(), 'supported_files': fields.Raw()
})
message_model = api.model('MessageInfo', {
    'id': fields.String(required=True), 'verzoeknummer': fields.String(), 'activity_name': fields.String(),
    'message_type': fields.String(), 'created_at': fields.DateTime(), 'original_filename': fields.String(),
    'human_verification_status': fields.String(), 'final_decision': fields.String(), 'validation_count': fields.Integer()
})
store_message_response = api.model('StoreMessageResponse', {
    'success': fields.Boolean(required=True), 'message_id': fields.String(required=True),
    'message': fields.String(required=True), 'parsed_data': fields.Raw()
})
error_response_model = api.model('ErrorResponse', {
    'success': fields.Boolean(required=True), 'error': fields.String(required=True),
    'decision': fields.String(), 'technical_reasons': fields.String(), 'explanation': fields.String()
})
business_rule_model = api.model('BusinessRule', {
    'id': fields.String(required=True), 'name': fields.String(required=True), 'version': fields.String(required=True),
    'file_name': fields.String(required=True), 'file_size': fields.Integer(), 'description': fields.String(),
    'is_active': fields.Boolean(required=True), 'created_at': fields.DateTime(), 'file_url': fields.String()
})
store_rule_response = api.model('StoreRuleResponse', {
    'success': fields.Boolean(required=True), 'rule_id': fields.String(required=True),
    'message': fields.String(required=True), 'rule_info': fields.Raw()
})
human_verification_response = api.model('HumanVerificationResponse', {
    'success': fields.Boolean(required=True), 'message': fields.String(required=True),
    'previous_ai_decision': fields.String(), 'human_decision': fields.String(), 'is_override': fields.Boolean()
})

# --- Helper Functions ---
def allowed_file(filename, file_type):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[file_type]

def get_file_extension(filename):
    if not filename: return '.tmp'
    return '.' + filename.rsplit('.', 1)[1].lower() if '.' in filename else '.tmp'

def create_storage_bucket_if_not_exists():
    """Create storage bucket for business rules if it doesn't exist"""
    if not supabase_client:
        return False
    
    try:
        # Try to list buckets to see if our bucket exists
        buckets = supabase_client.storage.list_buckets()
        bucket_names = [bucket.name for bucket in buckets]
        
        if STORAGE_BUCKET not in bucket_names:
            print(f"Creating storage bucket: {STORAGE_BUCKET}")
            supabase_client.storage.create_bucket(STORAGE_BUCKET, {"public": False})  # Changed to private
            print(f"Storage bucket {STORAGE_BUCKET} created successfully (private)")
        
        return True
    except Exception as e:
        print(f"Error with storage bucket: {e}")
        return False

# Load business rules from Supabase Storage
def load_business_rules_from_storage(rule_id):
    """Load business rules from Supabase Storage and create vector store"""
    if not supabase_client:
        raise Exception("Supabase not configured")
    
    # Get rule metadata from database
    result = supabase_client.table('business_rules').select('*').eq('id', rule_id).execute()
    if not result.data:
        raise Exception(f"Business rule with ID {rule_id} not found")
    
    rule = result.data[0]
    storage_path = rule['storage_path']
    file_name = rule['file_name']
    
    print(f"Loading business rules from storage: {file_name}")
    
    try:
        # Download file from Supabase Storage
        file_content = supabase_client.storage.from_(STORAGE_BUCKET).download(storage_path)
        
        # Create temporary file with the content
        file_extension = get_file_extension(file_name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        try:
            vectordb = load_business_rules(tmp_file_path)
            return vectordb
        finally:
            os.unlink(tmp_file_path)
            
    except Exception as e:
        print(f"Error downloading file from storage: {e}")
        raise Exception(f"Failed to load business rules from storage: {e}")

# --- Core Functions (No Heavy AI Models) ---
def load_business_rules(file_path):
    print(f"Loading business rules from: {file_path}")
    if not os.path.exists(file_path): 
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = file_path.split('.')[-1].lower()
    try:
        if ext == 'pdf': 
            loader = PyPDFLoader(file_path)
        elif ext in ['docx', 'doc']: 
            loader = Docx2txtLoader(file_path)
        else: 
            loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()
    except:
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=400, 
        separators=["Activiteit:", "Artikel", "§", "\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)
    print("Vector database created successfully")
    return vectordb

def read_xml_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()

def parse_xml_message(xml_content):
    try:
        root = ET.fromstring(xml_content)
        ns = {'vx': 'http://www.omgevingswet.nl/koppelvlak/stam-v4/verzoek'}
        
        verzoek_elem = root.find('.//vx:verzoeknummer', ns)
        verzoeknr = verzoek_elem.text if verzoek_elem is not None else None
        
        act_elems = root.findall('.//vx:activiteitnaam', ns)
        act_name = act_elems[0].text if act_elems else "Unknown Activity"
        
        type_elems = root.findall('.//vx:type', ns)
        msg_type = type_elems[0].text if type_elems else "Unknown Type"
        
        specs = []
        for s in root.findall('.//vx:specificatie', ns):
            answer_elem = s.find('vx:antwoord', ns)
            question_elem = s.find('vx:vraagtekst', ns)
            if answer_elem is not None and answer_elem.text:
                specs.append({
                    'question': question_elem.text if question_elem is not None else None,
                    'answer': answer_elem.text
                })
        
        attachments = []
        for d in root.findall('.//vx:document', ns):
            filename_elem = d.find('.//vx:bestandsnaam', ns)
            if filename_elem is not None and filename_elem.text:
                attachments.append(filename_elem.text)
        
        has_coords = len(root.findall('.//vx:coordinatenEtrs', ns)) > 0
        
        return {
            'activity_name': act_name, 'message_type': msg_type, 'specifications': specs, 
            'attachments': attachments, 'has_coordinates': has_coords, 'verzoeknummer': verzoeknr
        }
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return None

def retrieve_business_rules(vectordb, activity_name, message_type):
    queries = [
        f"indieningsvereisten voor activiteit {activity_name} type {message_type}", 
        f"Artikel {activity_name} {message_type}"
    ]
    docs = []
    for q in queries:
        print(f"Searching for: {q}")
        docs.extend(vectordb.similarity_search(q, k=5))
    
    unique_docs = {doc.page_content: doc for doc in docs}.values()
    filtered_docs = []
    type_kw = 'informatieplicht' if message_type.lower() == 'informatie' else 'melding'
    
    for doc in unique_docs:
        content = doc.page_content.lower()
        if activity_name.lower() in content and type_kw in content:
            if "graven in de bodem" in content and activity_name.lower() != "graven in de bodem": 
                continue
            filtered_docs.append(doc)
    
    if not filtered_docs:
        print("No highly relevant chunks found, using fallback.")
        return list(unique_docs)[:3]
    
    print(f"Filtered to {len(filtered_docs)} relevant chunks")
    return filtered_docs

def validate_message(message_data, rule_documents):
    report = {"checked_requirements": [], "checked_attachments": []}
    is_post_activity = any("evaluatieverslag" in s['answer'].lower() for s in message_data.get('specifications', []))
    print(f"Context: Post-activity submission detected: {is_post_activity}")
    type_kw = 'informatieplicht' if message_data['message_type'] == 'Informatie' else 'melding'

    for doc in rule_documents:
        content = doc.page_content.lower()
        if type_kw not in content: continue
        if "vallen niet" in content or "is niet van toepassing" in content: continue
        if is_post_activity and "voor het begin van de activiteit" in content: continue
        
        matches = re.finditer(r"^\s*([a-z0-9][°\.]\s+)(.+)", doc.page_content, re.MULTILINE)
        for match in matches:
            req_text = match.group(2).strip().replace('\n', ' ')
            if any(r['requirement'] == req_text for r in report['checked_requirements']): 
                continue
            
            found, evidence = False, "Informatie niet gevonden in XML."
            if "begrenzing" in req_text.lower() or "coördinaten" in req_text.lower():
                if message_data.get('has_coordinates'):
                    found, evidence = True, "Coördinaten aanwezig in bericht."
            elif "datum" in req_text.lower():
                for s in message_data['specifications']:
                    if 'datum' in s.get('question', '').lower() or 'datum' in s.get('answer', '').lower():
                        found, evidence = True, f"Gevonden in specificatie: '{s.get('answer')}'"
                        break
            else:
                keywords = set(re.findall(r'\b\w{4,}\b', req_text.lower()))
                for s in message_data['specifications']:
                    words = set(re.findall(r'\b\w{4,}\b', (s.get('answer', '') + s.get('question', '')).lower()))
                    if len(keywords.intersection(words)) > 1:
                        found, evidence = True, f"Mogelijk match in specificatie: '{s.get('answer')}'"
                        break
            
            report["checked_requirements"].append({
                "requirement": req_text, 
                "status": "MET" if found else "MISSING", 
                "reason": evidence, 
                "source_quote": doc.page_content
            })
    
    if is_post_activity:
        attached = len(message_data.get('attachments', [])) > 0
        report["checked_attachments"].append({
            "requirement": "Evaluatieverslag", 
            "status": "MET" if attached else "MISSING", 
            "reason": f"{len(message_data.get('attachments', []))} bijlagen gevonden." if attached else "Geen rapport bijgevoegd.", 
            "source_quote": "Afgeleid uit context van indiening."
        })
        is_valid = attached
    else:
        is_valid = not any(item['status'] == 'MISSING' for item in report['checked_requirements'])
    
    return {'is_valid': is_valid, 'details': report}

def generate_simple_explanation(validation_result, message_data):
    """Generate clear Dutch explanation without AI models"""
    activity = message_data.get('activity_name', 'Onbekende activiteit')
    message_type = message_data.get('message_type', 'Onbekend type')
    
    if validation_result['is_valid']:
        return f"Validatie geslaagd: Alle vereiste informatie voor '{activity}' ({message_type}) is aanwezig en compleet."
    else:
        missing_requirements = []
        missing_attachments = []
        
        for item in validation_result['details'].get('checked_requirements', []):
            if item['status'] == 'MISSING':
                missing_requirements.append(item['requirement'])
        
        for item in validation_result['details'].get('checked_attachments', []):
            if item['status'] == 'MISSING':
                missing_attachments.append(item['requirement'])
        
        explanation = f"Validatie gefaald voor '{activity}' ({message_type}).\n\n"
        
        if missing_requirements:
            explanation += f"Ontbrekende informatie:\n"
            for req in missing_requirements:
                explanation += f"• {req}\n"
        
        if missing_attachments:
            explanation += f"\nOntbrekende bijlagen:\n"
            for att in missing_attachments:
                explanation += f"• {att}\n"
        
        explanation += f"\nGelieve de ontbrekende informatie aan te vullen en opnieuw in te dienen."
        return explanation

def validate_xml_against_rules(xml_content, vectordb, rule_info=None):
    message_data = parse_xml_message(xml_content)
    if not message_data:
        return {
            "success": False, 
            "decision": "ERROR", 
            "summary_explanation": "Fout bij het verwerken van XML bericht.", 
            "validation_details": None
        }
    
    rule_docs = retrieve_business_rules(vectordb, message_data['activity_name'], message_data['message_type'])
    val_result = validate_message(message_data, rule_docs)
    summary = generate_simple_explanation(val_result, message_data)
    decision = "ACCEPTED" if val_result['is_valid'] else "REJECTED"
    
    response = {
        "success": True, 
        "decision": decision, 
        "summary_explanation": summary, 
        "validation_details": val_result['details']
    }
    
    if rule_info:
        response['rule_used'] = rule_info
    
    return response

# --- API Endpoint Parsers ---
message_store_parser = reqparse.RequestParser()
message_store_parser.add_argument('xml_file', location='files', type=FileStorage, required=True, help='XML message file')

validation_with_stored_rules_parser = reqparse.RequestParser()
validation_with_stored_rules_parser.add_argument('rule_id', type=str, required=False, help='Specific rule ID to use (optional, defaults to active rule)')

rules_upload_parser = reqparse.RequestParser()
rules_upload_parser.add_argument('rules_file', location='files', type=FileStorage, required=True, help='Business rules file')
rules_upload_parser.add_argument('name', type=str, required=True, help='Rule set name')
rules_upload_parser.add_argument('version', type=str, required=False, help='Version number (optional)')
rules_upload_parser.add_argument('description', type=str, required=False, help='Description (optional)')
rules_upload_parser.add_argument('make_active', type=bool, required=False, default=False, help='Make this rule active')

human_verification_parser = reqparse.RequestParser()
human_verification_parser.add_argument('verification_status', type=str, required=True, choices=['accepted', 'rejected'])
human_verification_parser.add_argument('verification_reason', type=str, required=True)
human_verification_parser.add_argument('verifier_name', type=str, required=True)

# --- SYSTEM ENDPOINTS ---
@ns_system.route('/health')
class HealthCheck(Resource):
    @ns_system.doc('health_check')
    @ns_system.marshal_with(health_response_model)
    def get(self):
        storage_ok = create_storage_bucket_if_not_exists()
        return {
            'status': 'healthy', 
            'supabase_connected': supabase_client is not None and storage_ok, 
            'message': 'Lightweight XML Validation API with Supabase Storage is running'
        }

@ns_system.route('/status')
class SystemStatus(Resource):
    @ns_system.doc('system_status')
    @ns_system.marshal_with(status_response_model)
    def get(self):
        storage_ok = create_storage_bucket_if_not_exists()
        return {
            'api_version': '1.6', 
            'status': 'running', 
            'supabase_status': {'connected': supabase_client is not None, 'storage': storage_ok}, 
            'supported_files': ALLOWED_EXTENSIONS
        }

# --- BUSINESS RULES MANAGEMENT ENDPOINTS ---
@ns_rules.route('/upload')
class BusinessRulesUpload(Resource):
    @ns_rules.doc('upload_business_rules')
    @ns_rules.expect(rules_upload_parser)
    @ns_rules.marshal_with(store_rule_response, code=200)
    def post(self):
        """Upload and store business rules file in Supabase Storage"""
        if not supabase_client:
            api.abort(500, "Supabase not configured")
        
        # Ensure storage bucket exists
        if not create_storage_bucket_if_not_exists():
            api.abort(500, "Failed to create or access storage bucket")
        
        args = rules_upload_parser.parse_args()
        rules_file = args['rules_file']
        name = args['name']
        version = args.get('version')
        description = args.get('description')
        make_active = args.get('make_active', False)
        
        if not rules_file or not allowed_file(rules_file.filename, 'rules'):
            api.abort(400, "Valid business rules file is required")
        
        try:
            # Read file content
            file_content = rules_file.read()
            
            # --- FIX: Sanitize the filename to remove invalid characters ---
            import re
            sanitized_filename = re.sub(r'[^a-zA-Z0-9._-]', '', rules_file.filename)
            # --- END OF FIX ---
            
            # Auto-generate version if not provided
            if not version:
                existing_rules = supabase_client.table('business_rules').select('version').eq('name', name).execute()
                if existing_rules.data:
                    versions = [float(rule.get('version', '1.0')) for rule in existing_rules.data]
                    max_version = max(versions) if versions else 1.0
                    version = f"{max_version + 0.1:.1f}"
                else:
                    version = "1.0"
            
            # Create unique storage path
            rule_id = str(uuid.uuid4())
            # Use the SANITIZED filename
            storage_path = f"rules/{rule_id}/{sanitized_filename}"
            
            # Upload file to Supabase Storage
            print(f"Uploading file to storage: {storage_path}")
            storage_response = supabase_client.storage.from_(STORAGE_BUCKET).upload(storage_path, file_content)
            
            # Get public URL for the file
            file_url = supabase_client.storage.from_(STORAGE_BUCKET).get_public_url(storage_path)
            
            # If making this rule active, deactivate others with same name
            if make_active:
                supabase_client.table('business_rules').update({'is_active': False}).eq('name', name).execute()
            
            # Store metadata in database (no file content!)
            rule_data = {
                'id': rule_id,
                'name': name,
                'version': version,
                # Also use the SANITIZED filename here for consistency
                'file_name': sanitized_filename,
                'file_size': len(file_content),
                'storage_path': storage_path,
                'file_url': file_url,
                # Use sanitized filename in description fallback
                'description': description or f"Business rules from {sanitized_filename}",
                'is_active': make_active
            }
            
            result = supabase_client.table('business_rules').insert(rule_data).execute()
            
            if result.data:
                rule_info = {
                    'id': rule_id,
                    'name': name,
                    'version': version,
                    'file_name': sanitized_filename,
                    'is_active': make_active,
                    'file_url': file_url
                }
                return {
                    'success': True,
                    'rule_id': rule_id,
                    'message': 'Business rules uploaded to Supabase Storage successfully',
                    'rule_info': rule_info
                }, 200
            else:
                # Clean up storage if database insert failed
                try:
                    supabase_client.storage.from_(STORAGE_BUCKET).remove([storage_path])
                except:
                    pass
                api.abort(500, "Failed to store business rules metadata")
                
        except Exception as e:
            print(f"Error uploading rules: {str(e)}")
            api.abort(500, f"Upload failed: {str(e)}")
            
@ns_rules.route('/list')
class BusinessRulesList(Resource):
    @ns_rules.doc('list_business_rules')
    @ns_rules.marshal_list_with(business_rule_model)
    def get(self):
        """Get all business rules with version history"""
        if not supabase_client:
            api.abort(500, "Supabase not configured")
        
        try:
            result = supabase_client.table('business_rules').select(
                'id, name, version, file_name, file_size, description, is_active, created_at, file_url'
            ).order('name, created_at', desc=True).execute()
            
            return result.data, 200
            
        except Exception as e:
            print(f"Error fetching rules: {str(e)}")
            api.abort(500, f"Failed to fetch rules: {str(e)}")

@ns_rules.route('/active')
class ActiveBusinessRules(Resource):
    @ns_rules.doc('get_active_rules')
    @ns_rules.marshal_list_with(business_rule_model)
    def get(self):
        """Get currently active business rules"""
        if not supabase_client:
            api.abort(500, "Supabase not configured")
        
        try:
            result = supabase_client.table('business_rules').select(
                'id, name, version, file_name, description, created_at, file_url'
            ).eq('is_active', True).execute()
            
            return result.data, 200
            
        except Exception as e:
            print(f"Error fetching active rules: {str(e)}")
            api.abort(500, f"Failed to fetch active rules: {str(e)}")

@ns_rules.route('/activate/<string:rule_id>')
class ActivateRule(Resource):
    @ns_rules.doc('activate_business_rule')
    def patch(self, rule_id):
        """Make a specific rule version active"""
        if not supabase_client:
            api.abort(500, "Supabase not configured")
        
        try:
            rule_result = supabase_client.table('business_rules').select('name').eq('id', rule_id).execute()
            if not rule_result.data:
                api.abort(404, "Rule not found")
            
            rule_name = rule_result.data[0]['name']
            
            # Deactivate other versions of same rule
            supabase_client.table('business_rules').update({'is_active': False}).eq('name', rule_name).execute()
            
            # Activate this version
            result = supabase_client.table('business_rules').update({'is_active': True}).eq('id', rule_id).execute()
            
            if result.data:
                return {'success': True, 'message': f'Rule {rule_id} activated successfully'}, 200
            else:
                api.abort(500, "Failed to activate rule")
                
        except Exception as e:
            print(f"Error activating rule: {str(e)}")
            api.abort(500, f"Activation failed: {str(e)}")

# --- MESSAGE ENDPOINTS ---
@ns_validation.route('/messages')
class MessageOperations(Resource):
    @ns_validation.doc('store_xml_message')
    @ns_validation.expect(message_store_parser)
    @ns_validation.marshal_with(store_message_response)
    def post(self):
        args = message_store_parser.parse_args()
        xml_file = args['xml_file']
        if not xml_file or not allowed_file(xml_file.filename, 'xml'): 
            api.abort(400, "Valid XML file is required.")
        
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xml')
        try:
            xml_file.save(tmp_file)
            tmp_file.close()
            xml_content = read_xml_file(tmp_file.name)
        finally:
            os.unlink(tmp_file.name)
        
        parsed_data = parse_xml_message(xml_content)
        if not parsed_data or not supabase_client: 
            api.abort(500, "Failed to parse XML or Supabase not configured.")
        
        db_data = {
            'xml_content': xml_content, 
            'original_filename': xml_file.filename,
            'verzoeknummer': parsed_data.get('verzoeknummer'), 
            'activity_name': parsed_data.get('activity_name'),
            'message_type': parsed_data.get('message_type'), 
            'message_metadata': parsed_data 
        }
        
        result = supabase_client.table('xml_messages').insert(db_data).execute()
        if result.data:
            return {
                'success': True, 
                'message_id': result.data[0]['id'], 
                'message': 'XML message stored.', 
                'parsed_data': parsed_data
            }, 200
        api.abort(500, "Failed to store message.")

    @ns_validation.doc('list_stored_messages')
    @ns_validation.marshal_list_with(message_model)
    def get(self):
        if not supabase_client: api.abort(500, "Supabase not configured.")
        result = supabase_client.table('xml_messages').select(
            'id, verzoeknummer, activity_name, message_type, created_at, original_filename, human_verification_status, final_decision, validation_count'
        ).order('created_at', desc=True).execute()
        return result.data, 200

# --- VALIDATION ENDPOINT (NO FILE UPLOADS NEEDED) ---
@ns_validation.route('/validate/<string:message_id>')
class ValidateStoredMessage(Resource):
    @ns_validation.doc('validate_stored_message_with_storage')
    @ns_validation.expect(validation_with_stored_rules_parser)
    @ns_validation.marshal_with(lightweight_validation_response_model, code=200)
    def post(self, message_id):
        """Validate stored XML message using business rules from Supabase Storage"""
        if not supabase_client: api.abort(500, "Supabase not configured.")
        
        args = validation_with_stored_rules_parser.parse_args()
        rule_id = args.get('rule_id')
        
        # Get the XML message
        result = supabase_client.table('xml_messages').select('*').eq('id', message_id).execute()
        if not result.data: api.abort(404, f"Message with ID {message_id} not found.")
        
        message_record = result.data[0]
        xml_content = message_record['xml_content']
        
        # Get business rules to use
        if rule_id:
            rule_result = supabase_client.table('business_rules').select('*').eq('id', rule_id).execute()
            if not rule_result.data:
                api.abort(404, f"Business rule with ID {rule_id} not found")
            rule_used = rule_result.data[0]
        else:
            active_rules = supabase_client.table('business_rules').select('*').eq('is_active', True).limit(1).execute()
            if not active_rules.data:
                api.abort(404, "No active business rules found. Please upload and activate rules first.")
            rule_used = active_rules.data[0]
        
        try:
            # Load rules from Supabase Storage (not database!)
            rules_vectordb = load_business_rules_from_storage(rule_used['id'])
            
            rule_info = {
                'id': rule_used['id'],
                'name': rule_used['name'],
                'version': rule_used['version']
            }
            
            validation_response = validate_xml_against_rules(xml_content, rules_vectordb, rule_info)
            
            # Update message record
            update_data = {
                'last_validation_result': validation_response,
                'validation_count': message_record.get('validation_count', 0) + 1
            }
            supabase_client.table('xml_messages').update(update_data).eq('id', message_id).execute()

            # Save to validation history
            print("Saving validation event to validation_history table...")
            history_data = {
                'message_id': message_id,
                'decision': validation_response['decision'],
                'explanation': validation_response['summary_explanation'],
                'technical_reasons': json.dumps(validation_response['validation_details']),
                'business_rules_used': f"{rule_used['name']} v{rule_used['version']}"
            }
            supabase_client.table('validation_history').insert(history_data).execute()
            
            return validation_response, 200
            
        except Exception as e:
            traceback.print_exc()
            api.abort(500, f"Validation error: {e}")

# --- HUMAN VERIFICATION ENDPOINTS ---
@ns_validation.route('/human-verification/<string:message_id>')
class HumanVerification(Resource):
    @ns_validation.doc('submit_human_verification')
    @ns_validation.expect(human_verification_parser)
    @ns_validation.marshal_with(human_verification_response, code=200)
    def post(self, message_id):
        if not supabase_client: return {'success': False, 'error': 'Supabase not configured'}, 500
        try:
            args = human_verification_parser.parse_args()
            result = supabase_client.table('xml_messages').select('*').eq('id', message_id).execute()
            if not result.data: return {'success': False, 'error': 'Message not found'}, 404
            
            message = result.data[0]
            ai_decision = message.get('last_validation_result', {}).get('decision', '').upper()
            human_decision = args['verification_status'].upper()
            is_override = ai_decision != human_decision and ai_decision in ['ACCEPTED', 'REJECTED']
            
            update_data = {
                'human_verification_status': args['verification_status'], 
                'human_verification_reason': args['verification_reason'], 
                'human_verified_at': datetime.utcnow().isoformat(), 
                'human_verifier_name': args['verifier_name'], 
                'final_decision': args['verification_status']
            }
            supabase_client.table('xml_messages').update(update_data).eq('id', message_id).execute()
            
            return {
                'success': True, 
                'message': 'Verification recorded', 
                'previous_ai_decision': ai_decision, 
                'human_decision': human_decision, 
                'is_override': is_override
            }, 200
        except Exception as e: 
            return {'success': False, 'error': str(e)}, 500

# --- ERROR HANDLERS ---
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# Initialize storage bucket on startup
def initialize_storage():
    if create_storage_bucket_if_not_exists():
        print(f"Storage bucket '{STORAGE_BUCKET}' is ready")
    else:
        print(f"Warning: Could not initialize storage bucket '{STORAGE_BUCKET}'")

# --- MAIN APP EXECUTION ---
if __name__ == '__main__':
    print("Starting Lightweight XML Validation API with Supabase Storage...")
    print("Swagger UI available at: http://localhost:5000/docs/")
    print("Rule Management with Supabase Storage:")
    print("  - POST /api/v1/rules/upload           - Upload business rules to storage")
    print("  - GET  /api/v1/rules/list             - List all rules")
    print("  - GET  /api/v1/rules/active           - Get active rules")
    print("  - PATCH /api/v1/rules/activate/{id}   - Activate specific rule")
    print("Validation Endpoints:")
    print("  - POST /api/v1/validation/messages    - Store XML messages")
    print("  - POST /api/v1/validation/validate/{message_id} - Validate using storage")
    print("No heavy AI models loaded - fast startup and cheap hosting!")
    
    # Initialize storage on startup
    initialize_storage()
    
    app.run(host='0.0.0.0', port=5000, debug=True)