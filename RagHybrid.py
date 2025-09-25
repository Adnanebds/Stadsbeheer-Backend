# Complete Backend API with CORS, Swagger Documentation, Supabase Integration and Human Verification
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
import torch # Make sure torch is imported

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
    print("⚠️ WARNING: Supabase credentials not found in environment variables")
    print("Please set SUPABASE_URL and SUPABASE_KEY")
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
    version='1.3',
    title='Enhanced XML Validation API',
    description='API for validating XML messages against business rules with detailed, evidence-based responses and human verification.',
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
    except ImportError:
        print("📦 Installing pypdf...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])

    try:
        import pdfplumber
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

# --- Swagger Models (Omitted for brevity, they are correct in your file) ---
validation_detail_item_model = api.model('ValidationDetailItem', {
    'requirement': fields.String(), 'status': fields.String(enum=['MET', 'MISSING']),
    'reason': fields.String(), 'source_quote': fields.String()
})
validation_details_model = api.model('ValidationDetails', {
    'checked_requirements': fields.List(fields.Nested(validation_detail_item_model)),
    'checked_attachments': fields.List(fields.Nested(validation_detail_item_model))
})
enhanced_validation_response_model = api.model('EnhancedValidationResult', {
    'success': fields.Boolean(required=True), 'decision': fields.String(required=True, enum=['ACCEPTED', 'REJECTED', 'ERROR']),
    'summary_explanation': fields.String(required=True), 'validation_details': fields.Nested(validation_details_model),
    'found_rules_context': fields.List(fields.String)
})
health_response_model = api.model('HealthCheck', {
    'status': fields.String(required=True), 'model_loaded': fields.Boolean(required=True),
    'supabase_connected': fields.Boolean(required=True), 'message': fields.String(required=True)
})
status_response_model = api.model('SystemStatus', {
    'api_version': fields.String(required=True), 'status': fields.String(required=True),
    'model_status': fields.Raw(), 'supabase_status': fields.Raw(), 'supported_files': fields.Raw()
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
pending_verification_model = api.model('PendingVerification', {
    'id': fields.String(required=True), 'verzoeknummer': fields.String(), 'activity_name': fields.String(),
    'message_type': fields.String(), 'ai_decision': fields.String(), 'ai_explanation': fields.String(),
    'ai_technical_reasons': fields.String(), 'created_at': fields.DateTime(), 'validation_count': fields.Integer(),
    'original_filename': fields.String()
})
human_verification_response = api.model('HumanVerificationResponse', {
    'success': fields.Boolean(required=True), 'message': fields.String(required=True),
    'previous_ai_decision': fields.String(), 'human_decision': fields.String(), 'is_override': fields.Boolean()
})
verification_status_model = api.model('VerificationStatus', {
    'message_id': fields.String(required=True), 'ai_decision': fields.String(), 'ai_explanation': fields.String(),
    'ai_technical_reasons': fields.String(), 'human_verification_status': fields.String(),
    'human_verification_reason': fields.String(), 'human_verified_at': fields.DateTime(),
    'human_verifier_name': fields.String(), 'final_decision': fields.String(), 'is_override': fields.Boolean()
})
verification_stats_model = api.model('VerificationStats', {
    'total_validated': fields.Integer(), 'pending_human_verification': fields.Integer(),
    'human_accepted': fields.Integer(), 'human_rejected': fields.Integer(), 'ai_human_agreement': fields.Integer(),
    'human_overrides': fields.Integer(), 'override_rate': fields.Float(), 'ai_accuracy_rate': fields.Float()
})

# --- Helper Functions ---
def allowed_file(filename, file_type):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[file_type]

def get_file_extension(filename):
    if not filename: return '.tmp'
    return '.' + filename.rsplit('.', 1)[1].lower() if '.' in filename else '.tmp'

# --- AI Model and RAG Core Functions ---
def init_model():
    global generation_pipeline
    if generation_pipeline is None:
        print("🤖 Initializing AI model...")
        generation_pipeline = setup_gemma_model()
    return generation_pipeline

def setup_gemma_model():
    print("Setting up Gemma 2-2b model for response generation...")
    try:
        model_name = "google/gemma-2-2b-it"
        
        # --- DEFINITIVE FIX ---
        # The 'device_map' argument is buggy on some systems.
        # This new method avoids it entirely.
        device = "cpu"
        dtype = torch.float32 
        print(f"Loading model onto device '{device}' with dtype '{dtype}'.")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 1. Load the model without device_map
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype
        )
        
        # 2. Assign the device directly in the pipeline
        pipeline_instance = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device, # Specify device here instead of in the model
            max_length=8192,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        print("✅ Gemma 2-2b model loaded successfully")
        return pipeline_instance
        
    except Exception as e:
        print(f"❌ Error loading Gemma model: {e}. Falling back to a smaller model.")
        try:
            return pipeline("text-generation", model="distilgpt2", max_length=300, pad_token_id=50256)
        except Exception as fallback_error:
            print(f"❌ Fallback model also failed: {fallback_error}")
            return None

def load_business_rules(file_path):
    print(f"Loading business rules from: {file_path}")
    if not os.path.exists(file_path): raise FileNotFoundError(f"File not found: {file_path}")
    ext = file_path.split('.')[-1].lower()
    try:
        if ext == 'pdf': loader = PyPDFLoader(file_path)
        elif ext in ['docx', 'doc']: loader = Docx2txtLoader(file_path)
        else: loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()
    except:
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400, separators=["Activiteit:", "Artikel", "§", "\n\n", "\n", " ", ""])
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)
    print("✅ Vector database created successfully")
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
        specs = [{'question': s.find('vx:vraagtekst', ns).text, 'answer': s.find('vx:antwoord', ns).text} for s in root.findall('.//vx:specificatie', ns) if s.find('vx:antwoord', ns) is not None]
        attachments = [d.find('.//vx:bestandsnaam', ns).text for d in root.findall('.//vx:document', ns) if d.find('.//vx:bestandsnaam', ns) is not None]
        has_coords = len(root.findall('.//vx:coordinatenEtrs', ns)) > 0
        return {'activity_name': act_name, 'message_type': msg_type, 'specifications': specs, 'attachments': attachments, 'has_coordinates': has_coords, 'verzoeknummer': verzoeknr}
    except Exception as e:
        print(f"❌ Error parsing XML: {e}")
        return None

# --- CORRECTED VALIDATION LOGIC ---
def retrieve_business_rules(vectordb, activity_name, message_type):
    queries = [f"indieningsvereisten voor activiteit {activity_name} type {message_type}", f"Artikel {activity_name} {message_type}"]
    docs = []
    for q in queries:
        print(f"🔍 Searching for: {q}")
        docs.extend(vectordb.similarity_search(q, k=5))
    unique_docs = {doc.page_content: doc for doc in docs}.values()
    filtered_docs = []
    type_kw = 'informatieplicht' if message_type.lower() == 'informatie' else 'melding'
    for doc in unique_docs:
        content = doc.page_content.lower()
        if activity_name.lower() in content and type_kw in content:
            if "graven in de bodem" in content and activity_name.lower() != "graven in de bodem": continue
            filtered_docs.append(doc)
    if not filtered_docs:
        print("⚠️ No highly relevant chunks found, using fallback.")
        return list(unique_docs)[:3]
    print(f"✅ Filtered to {len(filtered_docs)} relevant chunks")
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
            if any(r['requirement'] == req_text for r in report['checked_requirements']): continue
            
            found, evidence = False, "Info not in XML."
            if "begrenzing" in req_text.lower() or "coördinaten" in req_text.lower():
                if message_data.get('has_coordinates'):
                    found, evidence = True, "Coordinates provided in message."
            elif "datum" in req_text.lower():
                for s in message_data['specifications']:
                    if 'datum' in s.get('question', '').lower() or 'datum' in s.get('answer', '').lower():
                        found, evidence = True, f"Found in spec: '{s.get('answer')}'"
                        break
            else:
                keywords = set(re.findall(r'\b\w{4,}\b', req_text.lower()))
                for s in message_data['specifications']:
                    words = set(re.findall(r'\b\w{4,}\b', (s.get('answer', '') + s.get('question', '')).lower()))
                    if len(keywords.intersection(words)) > 1:
                        found, evidence = True, f"Potential match in spec: '{s.get('answer')}'"
                        break
            report["checked_requirements"].append({"requirement": req_text, "status": "MET" if found else "MISSING", "reason": evidence, "source_quote": doc.page_content})
    
    if is_post_activity:
        attached = len(message_data.get('attachments', [])) > 0
        report["checked_attachments"].append({"requirement": "Evaluatieverslag", "status": "MET" if attached else "MISSING", "reason": f"{len(message_data.get('attachments', []))} attachments found." if attached else "No report attached.", "source_quote": "Inferred from submission context."})
        is_valid = attached
    else:
        is_valid = not any(item['status'] == 'MISSING' for item in report['checked_requirements'])
    return {'is_valid': is_valid, 'details': report}

def generate_explanation(generation_pipeline, validation_result, message_data):
    if not generation_pipeline: return "AI model not loaded."
    findings = ""
    for item in validation_result['details']['checked_requirements']:
        if item['status'] == 'MISSING': findings += f"- Missing Info: '{item['requirement']}'\n"
    for item in validation_result['details']['checked_attachments']:
        if item['status'] == 'MISSING': findings += f"- Missing Attachment: '{item['requirement']}'\n"
    if not findings: findings = "All required info appears to be present."
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
You are a Dutch government official assistant. Write a formal, clear, and concise validation summary in Dutch.
Based on the following validation findings for the activity '{message_data.get('activity_name')}' ({message_data.get('message_type')}), write a brief summary.
If rejected, state what is missing. If accepted, state the submission is complete.
Validation Findings:
{findings}
<|eot_id|><|start_header_id|>model<|end_header_id|>
**Validatie Samenvatting:**
"""
    try:
        print("🤖 Generating AI summary explanation...")
        result = generation_pipeline(prompt, max_new_tokens=200)
        explanation = result[0]['generated_text'].split("model<|end_header_id|>")[1].replace("**Validatie Samenvatting:**", "").strip()
        return explanation if explanation else "Kon geen samenvatting genereren."
    except Exception as e:
        print(f"❌ Error generating explanation: {e}")
        return "Fout bij genereren van samenvatting."

def validate_xml_against_rules(xml_content, vectordb, generation_pipeline):
    message_data = parse_xml_message(xml_content)
    if not message_data:
        return {"success": False, "decision": "ERROR", "summary_explanation": "Failed to parse XML.", "validation_details": None, "found_rules_context": []}
    rule_docs = retrieve_business_rules(vectordb, message_data['activity_name'], message_data['message_type'])
    val_result = validate_message(message_data, rule_docs)
    summary = generate_explanation(generation_pipeline, val_result, message_data)
    decision = "ACCEPTED" if val_result['is_valid'] else "REJECTED"
    return {"success": True, "decision": decision, "summary_explanation": summary, "validation_details": val_result['details'], "found_rules_context": [d.page_content for d in rule_docs]}

# --- API Endpoint Parsers and Routes (Omitted for brevity, they are correct in your file) ---
message_store_parser = reqparse.RequestParser()
message_store_parser.add_argument('xml_file', location='files', type=FileStorage, required=True, help='XML message file')
validation_with_id_parser = reqparse.RequestParser()
validation_with_id_parser.add_argument('rules_file', location='files', type=FileStorage, required=True, help='Business rules file')
file_upload_parser = reqparse.RequestParser()
file_upload_parser.add_argument('rules_file', location='files', type=FileStorage, required=True)
file_upload_parser.add_argument('xml_file', location='files', type=FileStorage, required=True)
human_verification_parser = reqparse.RequestParser()
human_verification_parser.add_argument('verification_status', type=str, required=True, choices=['accepted', 'rejected'])
human_verification_parser.add_argument('verification_reason', type=str, required=True)
human_verification_parser.add_argument('verifier_name', type=str, required=True)

@ns_system.route('/health')
class HealthCheck(Resource):
    @ns_system.doc('health_check')
    @ns_system.marshal_with(health_response_model)
    def get(self):
        return {'status': 'healthy', 'model_loaded': generation_pipeline is not None, 'supabase_connected': supabase_client is not None, 'message': 'API is running'}

@ns_system.route('/status')
class SystemStatus(Resource):
    @ns_system.doc('system_status')
    @ns_system.marshal_with(status_response_model)
    def get(self):
        return {'api_version': '1.3', 'status': 'running', 'model_status': {'loaded': generation_pipeline is not None}, 'supabase_status': {'connected': supabase_client is not None}, 'supported_files': ALLOWED_EXTENSIONS}

@ns_validation.route('/messages')
class MessageOperations(Resource):
    @ns_validation.doc('store_xml_message')
    @ns_validation.expect(message_store_parser)
    @ns_validation.marshal_with(store_message_response)
    def post(self):
        args = message_store_parser.parse_args()
        xml_file = args['xml_file']
        if not xml_file or not allowed_file(xml_file.filename, 'xml'): api.abort(400, "Valid XML file is required.")
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xml')
        try:
            xml_file.save(tmp_file); tmp_file.close()
            xml_content = read_xml_file(tmp_file.name)
        finally:
            os.unlink(tmp_file.name)
        parsed_data = parse_xml_message(xml_content)
        if not parsed_data or not supabase_client: api.abort(500, "Failed to parse XML or Supabase not configured.")
        db_data = {
            'xml_content': xml_content, 'original_filename': xml_file.filename,
            'verzoeknummer': parsed_data.get('verzoeknummer'), 'activity_name': parsed_data.get('activity_name'),
            'message_type': parsed_data.get('message_type'), 'message_metadata': parsed_data 
        }
        result = supabase_client.table('xml_messages').insert(db_data).execute()
        if result.data:
            return {'success': True, 'message_id': result.data[0]['id'], 'message': 'XML message stored.', 'parsed_data': parsed_data}, 200
        api.abort(500, "Failed to store message.")

    @ns_validation.doc('list_stored_messages')
    @ns_validation.marshal_list_with(message_model)
    def get(self):
        if not supabase_client: api.abort(500, "Supabase not configured.")
        result = supabase_client.table('xml_messages').select('id, verzoeknummer, activity_name, message_type, created_at, original_filename, human_verification_status, final_decision, validation_count').order('created_at', desc=True).execute()
        return result.data, 200

# In RagHybrid.py

@ns_validation.route('/validate/<string:message_id>')
class ValidateStoredMessage(Resource):
    @ns_validation.doc('validate_stored_message_enhanced')
    @ns_validation.expect(validation_with_id_parser)
    @ns_validation.marshal_with(enhanced_validation_response_model, code=200)
    def post(self, message_id):
        if not supabase_client: api.abort(500, "Supabase not configured.")
        args = validation_with_id_parser.parse_args()
        rules_file = args['rules_file']
        if not rules_file or not allowed_file(rules_file.filename, 'rules'): api.abort(400, "A valid rules file is required.")
        
        result = supabase_client.table('xml_messages').select('*').eq('id', message_id).single().execute()
        if not result.data: api.abort(404, f"Message with ID {message_id} not found.")
        
        message_record = result.data
        xml_content = message_record['xml_content']
        init_model()

        with tempfile.NamedTemporaryFile(delete=False, suffix=get_file_extension(rules_file.filename)) as tmp_rules:
            rules_file.save(tmp_rules.name)
            rules_path = tmp_rules.name
        try:
            rules_vectordb = load_business_rules(rules_path)
            validation_response = validate_xml_against_rules(xml_content, rules_vectordb, generation_pipeline)
            
            # Update the main message record
            update_data = {
                'last_validation_result': validation_response,
                'validation_count': message_record.get('validation_count', 0) + 1
            }
            supabase_client.table('xml_messages').update(update_data).eq('id', message_id).execute()

            # --- ADD THIS BLOCK BACK ---
            # Save this validation event to the audit trail
            print("✍️ Saving validation event to validation_history table...")
            history_data = {
                'message_id': message_id,
                'decision': validation_response['decision'],
                'explanation': validation_response['summary_explanation'], # Use the new summary
                'technical_reasons': json.dumps(validation_response['validation_details']), # Store rich details as JSON string
                'business_rules_used': rules_file.filename
            }
            supabase_client.table('validation_history').insert(history_data).execute()
            # --- END OF ADDED BLOCK ---
            
            return validation_response, 200
        except Exception as e:
            traceback.print_exc()
            api.abort(500, f"Validation error: {e}")
        finally:
            os.unlink(rules_path)

@ns_validation.route('/messages/<string:message_id>')
class MessageDetails(Resource):
    @ns_validation.doc('get_message_details')
    def get(self, message_id):
        if not supabase_client: return {'error': 'Supabase not configured'}, 500
        try:
            result = supabase_client.table('xml_messages').select('*').eq('id', message_id).execute()
            if not result.data: return {'error': 'Message not found'}, 404
            return {'message': result.data[0]}, 200
        except Exception as e:
            return {'error': str(e)}, 500

# --- Human Verification Routes (and other remaining routes are unchanged) ---
@ns_validation.route('/human-verification/pending')
class PendingVerifications(Resource):
    @ns_validation.doc('get_pending_verifications')
    @ns_validation.marshal_list_with(pending_verification_model)
    def get(self):
        if not supabase_client: return {'error': 'Supabase not configured'}, 500
        try:
            result = supabase_client.table('xml_messages').select('id, verzoeknummer, activity_name, message_type, created_at, validation_count, last_validation_result, original_filename').is_('human_verification_status', 'null').not_.is_('last_validation_result', 'null').order('created_at', desc=False).execute()
            items = []
            for item in result.data:
                last_res = item.get('last_validation_result', {})
                items.append({'id': item['id'], 'verzoeknummer': item.get('verzoeknummer'), 'activity_name': item.get('activity_name'), 'message_type': item.get('message_type'), 'ai_decision': last_res.get('decision'), 'ai_explanation': last_res.get('summary_explanation'), 'ai_technical_reasons': json.dumps(last_res.get('validation_details')), 'created_at': item['created_at'], 'validation_count': item['validation_count'], 'original_filename': item.get('original_filename')})
            return items, 200
        except Exception as e: return {'error': str(e)}, 500

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
            update_data = {'human_verification_status': args['verification_status'], 'human_verification_reason': args['verification_reason'], 'human_verified_at': datetime.utcnow().isoformat(), 'human_verifier_name': args['verifier_name'], 'final_decision': args['verification_status']}
            supabase_client.table('xml_messages').update(update_data).eq('id', message_id).execute()
            return {'success': True, 'message': 'Verification recorded', 'previous_ai_decision': ai_decision, 'human_decision': human_decision, 'is_override': is_override}, 200
        except Exception as e: return {'success': False, 'error': str(e)}, 500
    
    @ns_validation.doc('get_verification_status')
    @ns_validation.marshal_with(verification_status_model)
    def get(self, message_id):
        if not supabase_client: return {'error': 'Supabase not configured'}, 500
        try:
            result = supabase_client.table('xml_messages').select('*').eq('id', message_id).execute()
            if not result.data: return {'error': 'Message not found'}, 404
            message = result.data[0]
            last_validation = message.get('last_validation_result', {})
            response = {'message_id': message_id, 'ai_decision': last_validation.get('decision'), 'ai_explanation': last_validation.get('summary_explanation'), 'ai_technical_reasons': json.dumps(last_validation.get('validation_details')), 'human_verification_status': message.get('human_verification_status'), 'human_verification_reason': message.get('human_verification_reason'), 'human_verified_at': message.get('human_verified_at'), 'human_verifier_name': message.get('human_verifier_name'), 'final_decision': message.get('final_decision'), 'is_override': (message.get('human_verification_status') and last_validation.get('decision') and message.get('human_verification_status').upper() != last_validation.get('decision').upper())}
            return response, 200
        except Exception as e: return {'error': str(e)}, 500

@ns_validation.route('/human-verification/stats')
class VerificationStats(Resource):
    @ns_validation.doc('get_verification_stats')
    @ns_validation.marshal_with(verification_stats_model)
    def get(self):
        if not supabase_client: return {'error': 'Supabase not configured'}, 500
        try:
            all_messages = supabase_client.table('xml_messages').select('human_verification_status, last_validation_result').not_.is_('last_validation_result', 'null').execute()
            stats = {'total_validated': len(all_messages.data), 'pending_human_verification': 0, 'human_accepted': 0, 'human_rejected': 0, 'ai_human_agreement': 0, 'human_overrides': 0}
            verified_count = 0
            for msg in all_messages.data:
                status = msg.get('human_verification_status')
                ai_decision = msg.get('last_validation_result', {}).get('decision', '').upper()
                if not status: stats['pending_human_verification'] += 1
                else:
                    verified_count += 1
                    if status == 'accepted': stats['human_accepted'] += 1
                    else: stats['human_rejected'] += 1
                    if ai_decision and status.upper() == ai_decision: stats['ai_human_agreement'] += 1
                    elif ai_decision and status.upper() != ai_decision: stats['human_overrides'] += 1
            if verified_count > 0:
                stats['override_rate'] = round((stats['human_overrides'] / verified_count) * 100, 2)
                stats['ai_accuracy_rate'] = round((stats['ai_human_agreement'] / verified_count) * 100, 2)
            else:
                stats['override_rate'] = 0.0; stats['ai_accuracy_rate'] = 0.0
            return stats, 200
        except Exception as e: return {'error': str(e)}, 500

@ns_validation.route('/validate')
class ValidateMessage(Resource):
    @ns_validation.doc('validate_xml_message_legacy')
    @ns_validation.expect(file_upload_parser)
    @ns_validation.marshal_with(enhanced_validation_response_model, code=200)
    def post(self):
        args = file_upload_parser.parse_args()
        rules_file, xml_file = args['rules_file'], args['xml_file']
        if not all([rules_file, xml_file]) or not allowed_file(rules_file.filename, 'rules') or not allowed_file(xml_file.filename, 'xml'): api.abort(400, "Valid rules and XML files are required.")
        init_model()
        with tempfile.NamedTemporaryFile(delete=False, suffix=get_file_extension(rules_file.filename)) as tmp_rules, \
             tempfile.NamedTemporaryFile(delete=False, suffix='.xml') as tmp_xml:
            rules_file.save(tmp_rules.name); xml_file.save(tmp_xml.name)
            rules_path, xml_path = tmp_rules.name, tmp_xml.name
        try:
            rules_vectordb = load_business_rules(rules_path)
            xml_content = read_xml_file(xml_path)
            response = validate_xml_against_rules(xml_content, rules_vectordb, generation_pipeline)
            return response, 200
        except Exception as e:
            traceback.print_exc()
            api.abort(500, f"Validation error: {e}")
        finally:
            os.unlink(rules_path); os.unlink(xml_path)

# --- Error Handlers ---
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# --- Main App Execution ---
if __name__ == '__main__':
    print("🚀 Starting Enhanced XML Validation API Server...")
    print("📚 Swagger UI available at: http://localhost:5000/docs/")
    init_model()
    app.run(host='0.0.0.0', port=5000, debug=True)