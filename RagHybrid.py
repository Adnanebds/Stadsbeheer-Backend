# Complete Backend API - MERGED VERSION (v2.1)
# This file combines the Activity Filtering logic with ALL the original endpoints.

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
from langchain_community.vectorstores import SupabaseVectorStore
import uuid
from datetime import datetime
from typing import List, Dict, Any

# Supabase integration
from supabase import create_client, Client

# ============================================================================
# BUSINESS RULES MAPPING - VALIDATABLE ACTIVITIES (NEW LOGIC)
# ============================================================================
VALIDATABLE_ACTIVITIES = {
    'Windturbine': {
        'has_rules': True,
        'sources': ['Paragraaf 4.136 Bal', 'Artikel 22.61 OP'],
        'confidence': 'HIGH',
        'rule_type': 'Melding + Informatieplicht'
    },
    'Graven in bodem met een kwaliteit boven de interventiewaarde bodemkwaliteit': {
        'has_rules': True,
        'sources': ['Paragraaf 4.120 Bal'],
        'confidence': 'HIGH',
        'rule_type': 'Melding + Informatieplicht'
    },
    'Gesloten bodemenergiesysteem': {
        'has_rules': True,
        'sources': ['Paragraaf 4.1137 Bal'],
        'confidence': 'HIGH',
        'rule_type': 'Melding + Informatieplicht'
    },
    'Saneren van de bodem': {
        'has_rules': True,
        'sources': ['Paragraaf 3.2.23 Bal', 'Paragraaf 4.121 Bal'],
        'confidence': 'HIGH',
        'rule_type': 'Melding + Informatieplicht'
    },
    'Bodem saneren': {
        'has_rules': True,
        'sources': ['Paragraaf 3.2.23 Bal', 'Paragraaf 4.121 Bal'],
        'confidence': 'HIGH',
        'rule_type': 'Melding + Informatieplicht',
        'note': 'Same as "Saneren van de bodem"'
    },
    'Opslaan van kuilvoer of vaste bijvoedermiddelen': {
        'has_rules': True,
        'sources': ['Paragraaf 4.84 Bal'],
        'confidence': 'HIGH',
        'rule_type': 'Melding + Informatieplicht'
    },
    'Mestvergistingsinstallatie': {
        'has_rules': True,
        'sources': ['Paragraaf 4.88 Bal'],
        'confidence': 'HIGH',
        'rule_type': 'Melding'
    },
    'Opslaan van vaste mest, champost of dikke fractie': {
        'has_rules': True,
        'sources': ['Paragraaf 4.84 Bal'],
        'confidence': 'HIGH',
        'rule_type': 'Melding'
    },
}

def is_activity_validatable(activity_name: str) -> bool:
    """Check if an activity can be validated"""
    if not activity_name:
        return False
    return activity_name in VALIDATABLE_ACTIVITIES and VALIDATABLE_ACTIVITIES[activity_name]['has_rules']

def get_activity_validation_info(activity_name: str) -> Dict[str, Any]:
    """Get validation information for an activity"""
    if is_activity_validatable(activity_name):
        return VALIDATABLE_ACTIVITIES[activity_name]
    return None
# ============================================================================

# AFTER imports, BEFORE any function definitions
print("🔄 Loading embedding model (one-time startup)...")
EMBEDDINGS_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
print("✅ Embedding model loaded and cached!")

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
    version='2.1', # MERGED VERSION
    title='XML Validation API with Activity Filtering',
    description='API that validates XML messages, manages rules, and filters by validatable activities.',
    doc='/docs/',
    prefix='/api/v1'
)

# Create namespaces
ns_validation = api.namespace('validation', description='XML message validation operations')
ns_rules = api.namespace('rules', description='Business rules management operations')
ns_system = api.namespace('system', description='System status and health checks')
# === NEW NAMESPACE ADDED ===
ns_activities = api.namespace('activities', description='Activity validation info')

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

# --- Swagger Models (from old file) ---
validation_detail_item_model = api.model('ValidationDetailItem', {
    'requirement': fields.String(description='The requirement text'),
    'status': fields.String(enum=['MET', 'MISSING'], description='Whether requirement is met'),
    'reason': fields.String(description='Explanation for the status'),
    'source_quote': fields.String(description='Quote from the business rules document'),
    'source_page': fields.Integer(description='Page number in source document'),
    'source_file': fields.String(description='Filename of source document'),
    'rule_name': fields.String(description='Name of the business rule'),
    'rule_version': fields.String(description='Version of the business rule')
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
    # === NEW: Added fields for enriched message list ===
    # 'is_validatable': fields.Boolean(),
    # 'validation_info': fields.Raw()
    # Note: No need to add to model, GET /messages returns raw JSON list
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

# --- Helper Functions (from old file) ---
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
        buckets = supabase_client.storage.list_buckets()
        bucket_names = [bucket.name for bucket in buckets]
        
        if STORAGE_BUCKET not in bucket_names:
            print(f"Creating storage bucket: {STORAGE_BUCKET}")
            supabase_client.storage.create_bucket(STORAGE_BUCKET, {"public": False})
            print(f"Storage bucket {STORAGE_BUCKET} created successfully (private)")
        
        return True
    except Exception as e:
        print(f"Error with storage bucket: {e}")
        return False

# Load business rules from Supabase Storage
def load_business_rules_from_storage(rule_id):
    """Load business rules from Supabase vector store - reuses existing embeddings"""
    if not supabase_client:
        raise Exception("Supabase client not initialized")
    
    try:
        # Get rule metadata from database
        rule_result = supabase_client.table('business_rules').select('*').eq('id', rule_id).execute()
        if not rule_result.data:
            raise Exception(f"Rule {rule_id} not found in database")
        
        rule = rule_result.data[0]
        
        # Check if vectors already exist in database
        existing_vectors = supabase_client.table('documents').select('id').eq('metadata->>rule_id', rule_id).limit(1).execute()
        
        if existing_vectors.data:
            # Vectors already exist - just connect to them!
            print(f"✅ Reusing existing vectors for rule {rule_id}")
            embeddings = EMBEDDINGS_MODEL
            
            vector_store = SupabaseVectorStore(
                client=supabase_client,
                embedding=embeddings,
                table_name="documents",
                query_name="match_documents"
            )
            
            print(f"Connected to existing vector store for rule {rule_id}")
            return vector_store
        
        # Vectors don't exist - need to create them
        print(f"📥 Creating new vectors for rule {rule_id}")
        
        file_path = rule['file_url'].replace(f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/", "")
        
        # Download file from Supabase Storage
        print(f"Downloading rule file: {file_path}")
        file_bytes = supabase_client.storage.from_(STORAGE_BUCKET).download(file_path)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=get_file_extension(rule['file_name'])) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Load document based on file type
            file_extension = get_file_extension(rule['file_name'])
            if file_extension == '.pdf':
                loader = PyPDFLoader(tmp_path)
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(tmp_path)
            elif file_extension == '.txt':
                loader = TextLoader(tmp_path, encoding='utf-8')
            else:
                loader = UnstructuredFileLoader(tmp_path)
            
            documents = loader.load()
            print(f"Loaded {len(documents)} document(s)")
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            texts = text_splitter.split_documents(documents)
            print(f"Split into {len(texts)} chunks")
            
            # Add metadata
            for doc in texts:
                doc.metadata.update({
                    'rule_id': rule_id,
                    'rule_name': rule['name'],
                    'rule_version': rule['version'],
                    'file_name': rule['file_name']
                })
            
            # Initialize embeddings
            embeddings = EMBEDDINGS_MODEL
            
            # Create Supabase vector store (this stores embeddings in DB)
            vector_store = SupabaseVectorStore.from_documents(
                texts,
                embeddings,
                client=supabase_client,
                table_name="documents",
                query_name="match_documents"
            )
            
            print(f"✅ Created and stored {len(texts)} vectors in Supabase")
            return vector_store
            
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except Exception as e:
        print(f"Error loading business rules: {e}")
        traceback.print_exc()
        raise

# --- Core Functions (from old file) ---
def read_xml_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()

# === MODIFIED XML PARSER (to match new file) ===
# === MODIFIED XML PARSER (v3) ===
# === MODIFIED XML PARSER (v3) ===
# === MODIFIED XML PARSER (v3) ===
# === MODIFIED XML PARSER (v3) ===
# === MODIFIED XML PARSER (v3) ===
def extract_xml_info(xml_content: str) -> Dict[str, Any]:
    """Parse XML to get key information (v3: Added indienDatum and structured specs)"""
    try:
        root = ET.fromstring(xml_content)
        # Define namespaces
        ns = {
            'vx': 'http://www.omgevingswet.nl/koppelvlak/stam-v4/verzoek',
            'ow': 'http://www.omgevingswet.nl/imow/bestandsinformatie/2023/04',
            'gml': 'http://www.opengis.net/gml'
        }
        
        # Helper to find text
        def find_text(path, namespaces=ns):
            elem = root.find(path, namespaces)
            return elem.text if elem is not None else None

        # Extract data
        verzoeknr = find_text('.//vx:verzoeknummer')
        act_name = find_text('.//vx:activiteitnaam')
        msg_type = find_text('.//vx:type')
        indien_datum = find_text('.//vx:indienDatum') # <-- *** FIX 1: Added indienDatum ***
        project_name = find_text('.//ow:naam')
        
        # Find initiatiefnemer name (can be in different spots)
        initiatiefnemer_name = find_text('.//vx:initiatiefnemer/vx:naam')
        if not initiatiefnemer_name:
             # Check contactpersoon if initiatiefnemer is empty
             contact_naam = find_text('.//vx:contactpersoon/vx:naam')
             # Check if contactpersoon is linked to an organisation
             contact_org = find_text('.//vx:contactpersoon/vx:organisatieNaam')
             if contact_org and contact_naam:
                 initiatiefnemer_name = f"{contact_naam} ({contact_org})"
             else:
                 initiatiefnemer_name = contact_naam # This will be None if contact_naam is None

        bevoegd_gezag = find_text('.//vx:bevoegdGezagNaam')

        specs = []
        for s in root.findall('.//vx:specificatie', ns):
            answer_elem = s.find('vx:antwoord', ns)
            question_elem = s.find('vx:vraagtekst', ns)
            bijlage_elem = s.find('vx:gevraagdeBijlage', ns)
            
            # Store structured data for validation
            spec_data = {
                'question': question_elem.text if question_elem is not None else None,
                'question_id': s.find('vx:vraagId', ns).text if s.find('vx:vraagId', ns) is not None else None,
                'answer': answer_elem.text if answer_elem is not None else None,
                'vraagclassificatie': s.find('vx:vraagclassificatie', ns).text if s.find('vx:vraagclassificatie', ns) is not None else None,
                'groepering': s.find('vx:groepering', ns).text if s.find('vx:groepering', ns) is not None else None,
                'has_bijlage': bijlage_elem is not None and bijlage_elem.find('.//vx:document', ns) is not None,
                'bijlage_documenten': []
            }
            
            if spec_data['has_bijlage']:
                for doc in bijlage_elem.findall('.//vx:document', ns):
                    spec_data['bijlage_documenten'].append({
                        'bestandsnaam': doc.find('vx:bestandsnaam', ns).text if doc.find('vx:bestandsnaam', ns) is not None else None,
                        'documentId': doc.find('vx:documentId', ns).text if doc.find('vx:documentId', ns) is not None else None
                    })
            
            specs.append(spec_data)
        
        attachments = []
        # Get all unique attachment filenames
        for d in root.findall('.//vx:document', ns):
            filename_elem = d.find('.//vx:bestandsnaam', ns)
            if filename_elem is not None and filename_elem.text:
                if filename_elem.text not in attachments:
                    attachments.append(filename_elem.text)
        
        has_coords = len(root.findall('.//gml:posList', ns)) > 0
        has_adres = root.find('.//vx:adresaanduiding', ns) is not None

        return {
            'verzoeknummer': verzoeknr,
            'activity_name': act_name,
            'message_type': msg_type,
            'indienDatum': indien_datum, # <-- *** FIX 1: Added to return ***
            'initiatiefnemer_name': initiatiefnemer_name, # Parsed name
            'bevoegd_gezag': bevoegd_gezag,
            'specifications': specs, # Structured specifications
            'attachments': attachments, # List of all attachment filenames
            'has_coordinates': has_coords,
            'has_adres': has_adres, # <-- *** FIX 2: Added address check ***
        }
    except Exception as e:
        print(f"Error parsing XML: {e}")
        traceback.print_exc()
        return {'error': str(e)}
    
def parse_xml_message(xml_content):
    return extract_xml_info(xml_content)


def retrieve_business_rules(vectordb, activity_name: str, message_type: str) -> List:
    """
    CORRECTED v6: Properly handle single-keyword activities like Windturbine.
    """
    print(f"\n🔍 Searching for rules: Activity='{activity_name}', Type='{message_type}'")

    # Normalize
    activity_normalized = activity_name.lower().strip()
    type_keyword = 'informatieplicht' if message_type.lower() == 'informatie' else 'melding'

    # Extract KEY words from activity name
    stop_words = {'van', 'de', 'het', 'een', 'in', 'op', 'of', 'en', 'voor', 'aan', 'bij', 'met', 'naar'}
    activity_keywords = [
        word for word in activity_normalized.split()
        if len(word) > 3 and word not in stop_words
    ]
    # If no keywords > 3 letters, use the original name parts (handles short names)
    if not activity_keywords:
         activity_keywords = [word for word in activity_normalized.split() if word not in stop_words]


    print(f"   🔑 Key activity words: {activity_keywords}")
    print(f"   🎯 Full activity name: '{activity_name}'")

    # Build focused search queries
    queries = []

    # Use all keywords if available, otherwise just the activity name
    if activity_keywords:
        queries.append(" ".join(activity_keywords[:2])) # Use first two keywords
        if len(activity_keywords) > 2:
             queries.append(activity_keywords[0]) # Add first keyword separately for focus
    else:
         queries.append(activity_normalized) # Use full name if no keywords extracted

    # Add specific activity-related searches (Keep these if relevant)
    if 'toepassen' in activity_keywords:
        queries.extend([
            "toepassen grond baggerspecie landbodem",
            "functionele toepassing kubieke meter"
        ])
    elif 'graven' in activity_keywords:
        queries.append("graven bodem interventiewaarde")
    elif 'saneren' in activity_keywords:
        queries.append("saneren bodem saneringsaanpak")
    elif 'windturbine' in activity_keywords: # Added specific query for windturbine
         queries.append("windturbine geluid risico")

    # Add type-specific query
    queries.append(f"{type_keyword} bevat")

    all_docs = []
    seen_content = set()

    for query in list(set(queries)): # Use set to avoid duplicate queries
        print(f"   📝 Query: '{query}'")
        try:
             results = vectordb.similarity_search(query, k=8)
             for doc in results:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append(doc)
        except Exception as search_err:
             print(f"   ⚠️ Error during similarity search for query '{query}': {search_err}")


    print(f"   ✅ Found {len(all_docs)} unique document chunks after search")

    # ULTRA-STRICT filtering - must match EXACT activity
    filtered_docs = []
    wrong_activity_docs = []

    # Define EXACT wrong activity phrases to HARD REJECT
    wrong_activities_map = {
         # --- Keep existing entries ---
        'toepassen': [
            'graven in de bodem','saneren van de bodem','bodemenergiesysteem',
            'windturbine','kuilvoer','mestvergisting','artikel 4.1224',
            'artikel 4.1235','paragraaf 4.120','§ 4.121','saneringsaanpak'
        ],
        'graven': [
            'toepassen grond','saneren van de bodem','bodemenergiesysteem',
            'artikel 4.1235','saneringsaanpak', 'windturbine' # Added windturbine
        ],
        'saneren': [
            'toepassen grond','graven in de bodem','bodemenergiesysteem',
            'artikel 4.1224', 'windturbine' # Added windturbine
        ],
         # --- ADD Entry for Windturbine ---
         'windturbine': [
              'graven in de bodem', 'saneren van de bodem', 'toepassen grond',
              'bodemenergiesysteem', 'kuilvoer', 'mestvergisting'
         ]
    }

    # Determine which wrong activities to exclude
    wrong_activities = []
    current_main_activity = None
    for key_activity in wrong_activities_map.keys():
        if key_activity in activity_keywords:
            wrong_activities = wrong_activities_map.get(key_activity, [])
            current_main_activity = key_activity
            break
    # If no specific map entry, use a generic exclusion list? (optional)
    if not current_main_activity:
         print(f"   ℹ️ No specific exclusion map found for '{activity_name}', using broad checks.")
         # wrong_activities = ['graven', 'saneren', 'toepassen grond', 'bodemenergie'] # Example generic list


    print(f"   🚫 Excluding documents containing: {wrong_activities}")

    for doc in all_docs:
        content_lower = doc.page_content.lower()
        page_num = doc.metadata.get('page', '?')

        # HARD REJECTION: Check for wrong activities FIRST
        has_wrong_activity = any(wrong_act in content_lower for wrong_act in wrong_activities)

        if has_wrong_activity:
            wrong_activity_docs.append(doc)
            # print(f"   ❌ EXCLUDED (Page {page_num}): Contains wrong activity phrase")
            continue

        # --- CORRECTED LOGIC ---
        # Count activity keyword matches
        keyword_matches = sum(1 for kw in activity_keywords if kw in content_lower)

        # Check if it has requirements structure
        has_requirements = bool(re.search(r'^\s*([a-z]\.|\d+[°º]\.)\s+', content_lower, re.MULTILINE))

        # Check if it mentions correct type (melding/informatieplicht)
        has_type_keyword = type_keyword in content_lower

        # ADJUSTED minimum keywords required
        if len(activity_keywords) == 1:
            min_keywords_required = 1 # Require only 1 match if only 1 keyword
        elif 'toepassen' in activity_keywords:
             min_keywords_required = 3
        else:
            min_keywords_required = 2

        # --- ADJUSTED DECISION LOGIC ---
        included = False
        reason = ""

        # Primary condition: Enough keywords + correct type
        if keyword_matches >= min_keywords_required and has_type_keyword:
             included = True
             reason = f"{keyword_matches}/{len(activity_keywords)} keywords + '{type_keyword}'"

        # Alternative for single keyword: Keyword + requirements structure + type
        elif len(activity_keywords) == 1 and keyword_matches >= 1 and has_requirements and has_type_keyword:
             included = True
             reason = f"Single keyword + Requirements structure + '{type_keyword}'"

        # Fallback for multi-keywords: Allow fewer keywords if requirements structure is present
        elif len(activity_keywords) > 1 and keyword_matches >= max(1, min_keywords_required - 1) and has_requirements and has_type_keyword:
             included = True
             reason = f"{keyword_matches}/{len(activity_keywords)} keywords (reduced) + Requirements structure + '{type_keyword}'"


        if included:
            filtered_docs.append(doc)
            print(f"   ✅ INCLUDED (Page {page_num}): {reason}")
        # else:
            # print(f"   ⏭️   SKIPPED (Page {page_num}): keywords={keyword_matches}/{len(activity_keywords)}, type={has_type_keyword}, req={has_requirements}, min_req={min_keywords_required}")
    # --- END OF CORRECTIONS ---


    # If no relevant rules found, return warning
    if not filtered_docs:
        print(f"   ⚠️   WARNING: No specific rule documents FOUND for activity '{activity_name}' after filtering!")
        print(f"   ⚠️   Initial search found {len(all_docs)} candidates, but filtering removed them.")
        print(f"   ⚠️   Check filter logic, keyword matching, and rule file content/structure.")
        print(f"   ⚠️   Found rules for potentially wrong activities: {len(wrong_activity_docs)} chunks")
        # Return empty to trigger proper error handling in validate_xml_against_rules
        return []

    print(f"   🎯 Final filtered count: {len(filtered_docs)} relevant chunks\n")
    return filtered_docs
    
# --- CORRECTED/IMPROVED validate_message function (v7) ---
# --- CORRECTED/IMPROVED validate_message function (v8) ---
# --- CORRECTED/IMPROVED validate_message function (v8) ---
# --- CORRECTED/IMPROVED validate_message function (v9) ---
# --- CORRECTED/IMPROVED validate_message function (v9) ---
# --- CORRECTED/IMPROVED validate_message function (v9) ---
# --- CORRECTED/IMPROVED validate_message function (v9.1) ---
def validate_message(message_data: Dict, rule_documents: List, rule_info: Dict = None) -> Dict:
    """
    CORRECTED v9.1:
    - Fixed KeyError: 'field' by changing check['field'] to check['id'].
    - Adds strict context filtering *inside* the function to discard irrelevant rules
      (e.g., discard 'mest' rules when validating 'Windturbine').
    - Correctly maps 'het adres waarop' to the basic location check.
    - Correctly checks for 'initiatiefnemer_name' in basic checks.
    - Prevents false-positive 'MET' status on irrelevant rules.
    - Tracks checked requirements to avoid duplicates.
    """
    print(f"\n🔍 VALIDATING MESSAGE (v9.1)")
    activity_name = message_data.get('activity_name', 'UNKNOWN')
    message_type = message_data.get('message_type', 'UNKNOWN')
    print(f"   Activity: {activity_name}")
    print(f"   Type: {message_type}")
    print(f"   Processing {len(rule_documents)} potentially relevant rule document chunks\n")

    report = {"checked_requirements": [], "checked_attachments": []}
    passed_basic_checks = set() 
    all_found_requirements_text = set() 

    # --- 1. Basic XML Field Checks (Corrected) ---
    print("📝 Checking basic XML fields...")
    basic_req_names = {
        'activity': 'Aanduiding van de activiteit',
        'date': 'Indieningsdatum (dagtekening)',
        'location': 'Locatie (adres of coördinaten)',
        'name': 'Naam en adres van indiener' # Check for initiator name
    }
    
    basic_checks = [
        {'id': 'activity', 'name': basic_req_names['activity'], 'present': bool(activity_name != 'UNKNOWN' and activity_name is not None)},
        {'id': 'date', 'name': basic_req_names['date'], 'present': bool(message_data.get('indienDatum'))},
        {'id': 'location', 'name': basic_req_names['location'], 'present': message_data.get('has_coordinates', False) or message_data.get('has_adres', False)},
        {'id': 'name', 'name': basic_req_names['name'], 'present': bool(message_data.get('initiatiefnemer_name'))}
    ]

    for check in basic_checks:
        status = "MET" if check['present'] else "MISSING"
        
        # --- *** THIS IS THE FIX *** ---
        # Changed check['field'] to check['id'] in the line below
        reason = f"✅ Veld '{check['id']}' aanwezig in XML." if status == "MET" else f"❌ Veld '{check['id']}' ontbreekt of kon niet worden geparsed in XML."
        # --- *** END OF FIX *** ---

        if check['id'] == 'name' and status == 'MET':
             reason = f"✅ Naam indiener aanwezig: {message_data.get('initiatiefnemer_name')}"
        elif check['id'] == 'name' and status == 'MISSING':
             reason = "❌ Naam indiener (initiatiefnemer/contactpersoon) niet gevonden in XML."

        print(f"   {'✅ MET' if status == 'MET' else '❌ MISSING'}: {check['name']}")
        report["checked_requirements"].append({
            "requirement": check['name'],
            "status": status,
            "reason": reason,
            "source_quote": "Basis XML structuurvereiste",
            "source_page": None,
            **({'source_file': rule_info.get('file_name'), 'rule_name': rule_info.get('name'), 'rule_version': rule_info.get('version')} if rule_info else {})
        })
        if status == "MET":
             passed_basic_checks.add(check['name'])
             all_found_requirements_text.add(check['name'])


    # --- 2. Define Context Keywords for Filtering ---
    activity_context_keywords = {
        'Windturbine': ['windturbine', 'rotor', 'mast', 'geluidhinder', 'plaatsgebonden risico', '4.136', '22.61', '4.1352', '4.1353'],
        'Graven in bodem met een kwaliteit boven de interventiewaarde bodemkwaliteit': ['graven', 'interventiewaarde', 'grondverzet', 'bodemkwaliteit', '4.120'],
        'Gesloten bodemenergiesysteem': ['bodemenergiesysteem', 'boorvrije', 'verticale boring', 'warmtewisselaar', '4.1137'],
        'Saneren van de bodem': ['saneren', 'saneringsplan', 'nazorgplan', 'evaluatieverslag', 'bus-melding', '3.2.23', '4.121'],
        'Bodem saneren': ['saneren', 'saneringsplan', 'nazorgplan', 'evaluatieverslag', 'bus-melding', '3.2.23', '4.121'],
        'Opslaan van kuilvoer of vaste bijvoedermiddelen': ['kuilvoer', 'bijvoedermiddelen', 'perssap', 'sleufsilo', '4.84'],
        'Mestvergistingsinstallatie': ['mestvergisting', 'vergistingsgas', 'digestaat', 'fakkel', '4.88'],
        'Opslaan van vaste mest, champost of dikke fractie': ['vaste mest', 'champost', 'mestsap', 'mestplaat', '4.84'],
    }
    
    current_context_keys = activity_context_keywords.get(activity_name, [activity_name.lower()])
    other_context_keys = []
    for act, keys in activity_context_keywords.items():
        if act != activity_name:
            other_context_keys.extend(keys)
    other_context_keys = list(set(other_context_keys)) 

    print(f"\n📄 Processing relevant rule document chunks...")
    print(f"   ℹ️  Huidige context (moet bevatten): {current_context_keys}")
    
    for idx, doc in enumerate(rule_documents):
        content = doc.page_content
        content_lower = content.lower()
        page_num = doc.metadata.get('page', '?')
        rule_file_name = doc.metadata.get('file_name', rule_info.get('file_name') if rule_info else 'Unknown')

        print(f"\n   --- Chunk {idx + 1}/{len(rule_documents)} (Page: {page_num}, File: {rule_file_name}) ---")

        # --- Stricter Context Check (v9) ---
        is_general_rules = 'algemene gegevens' in content_lower
        is_relevant_context = any(key in content_lower for key in current_context_keys)
        
        if not is_relevant_context and not is_general_rules:
            print(f"   ⏭️   Skipping chunk: Bevat geen keywords voor '{activity_name}' (bv. {current_context_keys}) of 'algemene gegevens'.")
            continue
            
        is_cross_contaminated = False
        if not is_general_rules: 
            for other_key in other_context_keys:
                if other_key in content_lower:
                    is_cross_contaminated = True
                    print(f"   ⏭️   Skipping chunk: Lijkt over een ANDERE activiteit te gaan (bevat keyword '{other_key}').")
                    break
        
        if is_cross_contaminated:
            continue
        
        print(f"   ℹ️  Chunk is relevant voor '{activity_name}' of 'algemene gegevens'.")

        # Extract requirements
        patterns = [
            r'^\s*([a-z]\.)\s+(.+?)(?=\n\s*[a-z]\.|\n\s*$|\n\n)', 
            r'^\s*(\d+[°º]\.)\s+(.+?)(?=\n\s*\d+[°º]\.|\n\s*$|\n\n)',
        ]
        requirements_in_chunk = []
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
            for match in matches:
                req_text = ' '.join(match.group(2).strip().split())
                if len(req_text) > 10 and req_text not in all_found_requirements_text:
                     requirements_in_chunk.append(req_text)
                     all_found_requirements_text.add(req_text)

        if not requirements_in_chunk:
             print(f"   ℹ️ Geen gestructureerde requirements (a./1°.) gevonden in deze chunk.")
             continue

        print(f"   📋 Found {len(requirements_in_chunk)} unique new requirements in chunk.")

        # --- 3. Validate Each Requirement Found in Chunk ---
        for req_text in requirements_in_chunk:
            req_text_lower = req_text.lower()
            print(f"      ➡️   Requirement: '{req_text[:100]}...'")

            found_status = "MISSING"
            evidence = f"❌ Informatie voor '{req_text[:50]}...' niet gevonden in XML specificaties of bijlagen."

            # --- Check if this requirement was already covered by Basic Checks ---
            if 'aanduiding van de activiteit' in req_text_lower and basic_req_names['activity'] in passed_basic_checks:
                found_status = "MET"
                evidence = "✅ Voldoet aan basis XML vereiste (activity_name)."
            elif 'dagtekening' in req_text_lower and basic_req_names['date'] in passed_basic_checks:
                found_status = "MET"
                evidence = "✅ Voldoet aan basis XML vereiste (indienDatum)."
            elif any(kw in req_text_lower for kw in ['locatie', 'kadastraal', 'coördinaten', 'adres waarop']) and basic_req_names['location'] in passed_basic_checks:
                 found_status = "MET"
                 evidence = "✅ Voldoet aan basis XML vereiste (has_coordinates of has_adres)."
            elif any(kw in req_text_lower for kw in ['naam en het adres', 'degene die de activiteit verricht']) and basic_req_names['name'] in passed_basic_checks:
                 found_status = "MET"
                 evidence = f"✅ Voldoet aan basis XML vereiste: {message_data.get('initiatiefnemer_name')}"
            
            # --- If not covered by basic checks, run specific checks ---
            
            elif any(kw in req_text_lower for kw in ['naam en het adres', 'degene die de activiteit verricht']):
                 if basic_req_names['name'] not in passed_basic_checks: 
                    evidence = "❌ Naam en adres van indiener (initiatiefnemer/contactpersoon) niet gevonden in XML."
                    found_status = "MISSING"

            # --- Windturbine Specific Checks (MUST come before general location/name checks) ---
            elif 'windturbine' in activity_name.lower():
                if 'situatietekening' in req_text_lower: # (art 4.1352 a)
                    if any('situatietekening' in att.lower() for att in message_data.get('attachments', [])):
                        found_status = "MET"
                        evidence = "✅ Locatie waarschijnlijk aanwezig in bijlage 'Situatietekening'."
                    elif basic_req_names['location'] in passed_basic_checks:
                        found_status = "MET" 
                        evidence = "✅ Locatie/coördinaten/adres aanwezig in XML (als alternatief voor situatietekening)."
                    else:
                        evidence = "❌ Situatietekening (of locatie/adres/coördinaten) ontbreekt."

                elif 'akoestisch onderzoek' in req_text_lower: # (art 4.1352 b)
                    if any('akoestisch' in att.lower() for att in message_data.get('attachments', [])):
                        found_status = "MET"
                        evidence = "✅ Akoestisch onderzoek waarschijnlijk aanwezig in bijlage."
                    else:
                        found_status = "N/A" 
                        evidence = "ℹ️ Akoestisch onderzoek mogelijk niet vereist (bv. < 15m) of niet gevonden als bijlage."

                elif any(kw in req_text_lower for kw in ['afstand tot gebouwen', 'afstand tot hoogspanningslijnen', 'plaatsgebonden risico']): # (art 4.1353 a,b)
                    if any(spec.get('answer') == 'true' and spec.get('has_bijlage') and 'plaatsgebonden risico' in spec.get('question','').lower() for spec in message_data.get('specifications', [])):
                        found_status = "MET"
                        evidence = "✅ Bijlage voor 'plaatsgebonden risico' is aangevinkt in XML."
                    elif any(kw in att.lower() for att in message_data.get('attachments', []) for kw in ['veiligheid', 'risico', 'afstand', 'rekenbestanden']):
                        found_status = "MET"
                        evidence = "✅ Veiligheidsinfo/afstanden waarschijnlijk aanwezig in bijlage."
                    else:
                        evidence = f"❌ Veiligheidsinformatie (afstanden/risico) niet gevonden in bijlagen."
                
                elif any(kw in req_text_lower for kw in ['technische gegevens', 'rotor', 'diameter', 'mast', 'hoogte']): # (art 4.1353 c,d)
                    if any(spec.get('answer') == 'true' and spec.get('has_bijlage') and 'gegevens windturbine' in spec.get('question','').lower() for spec in message_data.get('specifications', [])):
                        found_status = "MET"
                        evidence = "✅ Bijlage voor 'Gegevens windturbine' is aangevinkt in XML."
                    elif any(kw in att.lower() for att in message_data.get('attachments', []) for kw in ['technisch', 'specificatie', 'offerte', 'gegevens']):
                        found_status = "MET"
                        evidence = f"✅ Technische gegevens mogelijk aanwezig in algemene bijlage(n)."
                    else:
                        evidence = f"❌ Technische gegevens niet gevonden in bijlagen."

            # --- Prevent false 'MET' on irrelevant rules (like vergistingsgas + coördinaten) ---
            if found_status == "MET":
                # Check if this 'MET' status came from a basic check (e.g., location)
                basic_match = False
                if any(kw in req_text_lower for kw in ['locatie', 'kadastraal', 'coördinaten', 'adres waarop']) and basic_req_names['location'] in passed_basic_checks:
                    basic_match = True
                
                if basic_match:
                    # Now check if this rule *also* contains irrelevant keywords
                    is_irrelevant = False
                    for other_key in other_context_keys:
                        if other_key in req_text_lower:
                            is_irrelevant = True
                            break
                    
                    if is_irrelevant:
                        print(f"   ⚠️   Skipping false positive MET for irrelevant rule: {req_text[:50]}...")
                        all_found_requirements_text.remove(req_text) # Allow re-checking if it appears in a valid chunk
                        continue # Don't add this false positive to the report

            print(f"         Status: {found_status} - Reason: {evidence}")

            report["checked_requirements"].append({
                "requirement": req_text,
                "status": found_status,
                "reason": evidence,
                "source_quote": content[:400],
                "source_page": page_num,
                **({'source_file': rule_info.get('file_name'), 'rule_name': rule_info.get('name'), 'rule_version': rule_info.get('version')} if rule_info else {})
            })

    # --- 4. Final Decision ---
    missing_requirements = [r for r in report['checked_requirements'] if r['status'] == 'MISSING']
    is_valid = len(missing_requirements) == 0

    total_req_checked = len(report['checked_requirements'])
    met_req = sum(1 for r in report['checked_requirements'] if r['status'] == 'MET')
    missing_req_count = len(missing_requirements)
    na_req_count = sum(1 for r in report['checked_requirements'] if r['status'] == 'N/A')

    print(f"\n{'='*60}")
    print(f"📊 VALIDATION SUMMARY for '{activity_name}' (v9.1)")
    print(f"{'='*60}")
    print(f"Total requirements checked (incl. basic): {total_req_checked}")
    print(f"Requirements MET: {met_req}")
    print(f"Requirements MISSING: {missing_req_count}")
    print(f"Requirements N/A (Not Applicable/Checked): {na_req_count}")
    print(f"\nFinal decision: {'✅ VALID (ACCEPTED)' if is_valid else '❌ INVALID (REJECTED)'}")
    print(f"{'='*60}\n")
    
    return {
        'is_valid': is_valid,
        'details': report
    }
# --- CORRECTED generate_simple_explanation function (v3) ---
# --- CORRECTED generate_simple_explanation function (v3) ---
def generate_simple_explanation(validation_result: Dict, message_data: Dict) -> str:
    """
    Generate clear Dutch explanation - CORRECTED v3
    - Uses a set to avoid duplicate missing requirements in the summary.
    - Focuses only on items explicitly marked MISSING in the report.
    - Adds rule file info to the summary.
    """
    activity = message_data.get('activity_name', 'Onbekende activiteit')
    message_type = message_data.get('message_type', 'Onbekend type')
    rule_info = validation_result.get('rule_used')
    source_file_text = ""
    if rule_info and rule_info.get('file_name'):
         source_file_text = f" (Regelset: {rule_info.get('file_name', '')} v{rule_info.get('version', '')})"

    if validation_result.get('is_valid', False):
        return f"Validatie geslaagd{source_file_text}: Alle vereiste informatie voor '{activity}' ({message_type}) lijkt aanwezig te zijn op basis van de business rules en het XML-bericht."
    else:
        # Use a set to store unique missing requirements text
        missing_items_set = set() 

        # Add missing requirements from the detailed check
        for item in validation_result.get('details', {}).get('checked_requirements', []):
            if item.get('status') == 'MISSING':
                # Use the 'name' for basic checks, 'requirement' for rule-based checks
                missing_text = item.get('name') or item.get('requirement')
                if missing_text:
                     # Truncate long requirement texts for clarity in the summary
                     missing_items_set.add(missing_text[:150] + ('...' if len(missing_text) > 150 else ''))

        # Add missing attachments if that list is still used (though logic moved to requirements)
        for item in validation_result.get('details', {}).get('checked_attachments', []):
             if item.get('status') == 'MISSING':
                  missing_text = item.get('requirement')
                  if missing_text:
                       missing_items_set.add(missing_text[:150] + ('...' if len(missing_text) > 150 else ''))


        explanation = f"Validatie gefaald voor '{activity}' ({message_type}){source_file_text}.\n\n"

        if not missing_items_set:
             explanation += "Validatie is mislukt, maar er zijn geen specifieke ontbrekende punten gelogd. Controleer de technische details of de server logs."
        else:
            explanation += f"Mogelijk ontbrekende of onvolledige informatie:\n"
            # Sort for consistent output
            for req in sorted(list(missing_items_set)):
                explanation += f"• {req}\n"

        explanation += f"\nControleer de melding en de bijlagen tegen de indieningsvereisten. Vul eventueel aan en dien opnieuw in."
        return explanation    
def validate_xml_against_rules(xml_content, vectordb, rule_info=None):
    """
    (This is from your old file, unchanged)
    """
    message_data = parse_xml_message(xml_content)
    if not message_data or 'error' in message_data:
        return {
            "success": False, 
            "decision": "ERROR", 
            "summary_explanation": "Fout bij het verwerken van XML bericht.", 
            "validation_details": None
        }
    
    rule_docs = retrieve_business_rules(vectordb, message_data['activity_name'], message_data['message_type'])
    
    # Check if rules were found
    if not rule_docs:
        activity = message_data.get('activity_name', 'Onbekende activiteit')
        return {
            "success": False,
            "decision": "ERROR",
            "summary_explanation": f"Geen business rules gevonden voor activiteit '{activity}'. Het geüploade regelbestand bevat waarschijnlijk geen regels voor deze activiteit. Upload een regelbestand dat de juiste regels bevat.",
            "validation_details": {
                "checked_requirements": [],
                "checked_attachments": []
            }
        }
    
    val_result = validate_message(message_data, rule_docs, rule_info)
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


# --- API Endpoint Parsers (from old file) ---
message_store_parser = reqparse.RequestParser()
message_store_parser.add_argument('file', location='files', type=FileStorage, required=True, help='XML message file') # Renamed to 'file' to match new code

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

# --- SYSTEM ENDPOINTS (from old file) ---
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
            'api_version': '2.1', # MERGED VERSION
            'status': 'running', 
            'supabase_status': {'connected': supabase_client is not None, 'storage': storage_ok}, 
            'supported_files': ALLOWED_EXTENSIONS
        }

# --- BUSINESS RULES MANAGEMENT ENDPOINTS (from old file) ---
@ns_rules.route('/upload')
class BusinessRulesUpload(Resource):
    @ns_rules.doc('upload_business_rules')
    @ns_rules.expect(rules_upload_parser)
    @ns_rules.marshal_with(store_rule_response, code=200)
    def post(self):
        """Upload and store business rules file in Supabase Storage"""
        if not supabase_client:
            api.abort(500, "Supabase not configured")
        
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
            file_content = rules_file.read()
            
            import re
            sanitized_filename = re.sub(r'[^a-zA-Z0-9._-]', '', rules_file.filename)
            
            if not version:
                existing_rules = supabase_client.table('business_rules').select('version').eq('name', name).execute()
                if existing_rules.data:
                    versions = [float(rule.get('version', '1.0')) for rule in existing_rules.data]
                    max_version = max(versions) if versions else 1.0
                    version = f"{max_version + 0.1:.1f}"
                else:
                    version = "1.0"
            
            rule_id = str(uuid.uuid4())
            storage_path = f"rules/{rule_id}/{sanitized_filename}"
            
            print(f"Uploading file to storage: {storage_path}")
            
            mime_type = 'application/pdf' if sanitized_filename.endswith('.pdf') else \
                        'application/vnd.openxmlformats-officedocument.wordprocessingml.document' if sanitized_filename.endswith('.docx') else \
                        'text/plain'

            # Upload to storage
            supabase_client.storage.from_(STORAGE_BUCKET).upload(
                storage_path, 
                file_content,
                file_options={"content-type": mime_type}
            )
            
            file_url = supabase_client.storage.from_(STORAGE_BUCKET).get_public_url(storage_path)
            
            if make_active:
                supabase_client.table('business_rules').update({'is_active': False}).eq('name', name).execute()
            
            rule_data = {
                'id': rule_id,
                'name': name,
                'version': version,
                'file_name': sanitized_filename,
                'file_size': len(file_content),
                'storage_path': storage_path,
                'file_url': file_url,
                'description': description or f"Business rules from {sanitized_filename}",
                'is_active': make_active
            }
            
            result = supabase_client.table('business_rules').insert(rule_data).execute()
            
            if result.data:
                # 🚀 NEW: Create vectors immediately after upload
                print(f"🔄 Creating vectors for rule {rule_id}...")
                try:
                    # Save file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=get_file_extension(sanitized_filename)) as tmp_file:
                        tmp_file.write(file_content)
                        tmp_path = tmp_file.name
                    
                    try:
                        # Load document
                        file_extension = get_file_extension(sanitized_filename)
                        if file_extension == '.pdf':
                            loader = PyPDFLoader(tmp_path)
                        elif file_extension in ['.docx', '.doc']:
                            loader = Docx2txtLoader(tmp_path)
                        elif file_extension == '.txt':
                            loader = TextLoader(tmp_path, encoding='utf-8')
                        else:
                            loader = UnstructuredFileLoader(tmp_path)
                        
                        documents = loader.load()
                        print(f"📄 Loaded {len(documents)} document(s)")
                        
                        # Split documents
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200,
                            length_function=len
                        )
                        texts = text_splitter.split_documents(documents)
                        print(f"✂️ Split into {len(texts)} chunks")
                        
                        # Add metadata
                        for doc in texts:
                            doc.metadata.update({
                                'rule_id': rule_id,
                                'rule_name': name,
                                'rule_version': version,
                                'file_name': sanitized_filename
                            })
                        
                        # Initialize embeddings
                        embeddings = EMBEDDINGS_MODEL # Use pre-loaded model
                        
                        # Create vectors in Supabase
                        vector_store = SupabaseVectorStore.from_documents(
                            texts,
                            embeddings,
                            client=supabase_client,
                            table_name="documents",
                            query_name="match_documents"
                        )
                        
                        print(f"✅ Created {len(texts)} vectors in Supabase!")
                        
                    finally:
                        # Cleanup temp file
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    
                except Exception as vector_error:
                    print(f"⚠️ Warning: Failed to create vectors: {vector_error}")
                    # Don't fail the upload if vector creation fails
                    # Vectors will be created on first validation instead
                
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
                    'message': 'Business rules uploaded and vectors created successfully',
                    'rule_info': rule_info
                }, 200
            else:
                try:
                    supabase_client.storage.from_(STORAGE_BUCKET).remove([storage_path])
                except:
                    pass
                api.abort(500, "Failed to store business rules metadata")
                
        except Exception as e:
            print(f"Error uploading rules: {str(e)}")
            traceback.print_exc()
            api.abort(500, f"Upload failed: {str(e)}")

# === THIS IS ONE OF THE MISSING ENDPOINTS ===
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
            
            # Return raw data list
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
            
            supabase_client.table('business_rules').update({'is_active': False}).eq('name', rule_name).execute()
            
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
    
    # === NEW, MERGED POST METHOD ===
    @ns_validation.doc('store_xml_message')
    @ns_validation.expect(message_store_parser)
    # @ns_validation.marshal_with(store_message_response) # Cannot use marshal, custom 400 response
    def post(self):
        """Store XML message (ONLY if activity is validatable)"""
        if not supabase_client:
            api.abort(500, 'Supabase not configured')
        
        args = message_store_parser.parse_args()
        file = args['file']
        
        if not file:
            api.abort(400, 'No file provided')
        
        if file.filename == '':
            api.abort(400, 'Empty filename')
        
        if not allowed_file(file.filename, 'xml'):
            api.abort(400, 'Invalid file type. Only XML files allowed')
        
        try:
            # Read XML content
            xml_content = file.read().decode('utf-8')
            
            # Parse and extract info
            parsed_info = extract_xml_info(xml_content)
            activity_name = parsed_info.get('activity_name', 'UNKNOWN')
            
            # CHECK IF VALIDATABLE - This is the key filter!
            if not is_activity_validatable(activity_name):
                return {
                    'success': False,
                    'error': 'Activity not validatable',
                    'message': f'The activity "{activity_name}" does not have business rules configured and cannot be validated.',
                    'activity_name': activity_name,
                    'validatable_activities': list(VALIDATABLE_ACTIVITIES.keys())
                }, 400
            
            # Store message
            message_id = str(uuid.uuid4())
            message_data = {
                'id': message_id,
                'xml_content': xml_content,
                'verzoeknummer': parsed_info.get('verzoeknummer'),
                'activity_name': activity_name,
                'message_type': parsed_info.get('message_type'),
                'original_filename': file.filename,
                'message_metadata': parsed_info, # Store all parsed data
                'project_name': parsed_info.get('project_name'),
                'initiatiefnemer_name': parsed_info.get('initiatiefnemer_name'),
                'bevoegd_gezag': parsed_info.get('bevoegd_gezag')
            }
            
            supabase_client.table('xml_messages').insert(message_data).execute()
            
            return {
                'success': True,
                'message_id': message_id,
                'message': 'Message stored successfully',
                'parsed_data': parsed_info,
                'is_validatable': True
            }, 201
            
        except Exception as e:
            traceback.print_exc()
            api.abort(500, f'Error storing message: {str(e)}')

    # === NEW, MERGED GET METHOD ===
    @ns_validation.doc('list_stored_messages')
    @ns_validation.param('validatable_only', 'Filter to only validatable messages', type='boolean', default='true')
    @ns_validation.param('limit', 'Maximum messages to return', type='integer', default=50)
    # @ns_validation.marshal_list_with(message_model) # Cannot marshal, response is custom dict
    # === CORRECTED GET METHOD (v2.2) ===
    @ns_validation.doc('list_stored_messages')
    @ns_validation.param('validatable_only', 'Filter to only validatable messages', type='boolean', default='true')
    @ns_validation.param('limit', 'Maximum messages to return', type='integer', default=1000) # Increased default limit
    @ns_validation.param('offset', 'Number of messages to skip (for pagination)', type='integer', default=0)
    def get(self):
        """List messages (filtered by default to validatable only) with pagination"""
        if not supabase_client:
            api.abort(500, 'Supabase not configured')

        try:
            validatable_only = request.args.get('validatable_only', 'true').lower() == 'true'
            # Set a reasonable upper limit to prevent excessively large requests, e.g., 5000
            # The frontend requested 3608, which might be okay, but capping is safer.
            # Let's use 4000 as a cap, but allow the user request up to that.
            requested_limit = request.args.get('limit', 1000, type=int)
            limit = min(requested_limit, 4000) # Cap the limit
            offset = request.args.get('offset', 0, type=int)

            query = supabase_client.table('xml_messages').select(
                'id, verzoeknummer, activity_name, message_type, created_at, original_filename, human_verification_status, final_decision, validation_count, project_name, initiatiefnemer_name, bevoegd_gezag',
                count='exact' # Ask Supabase for the total count matching the query
            ).order('created_at', desc=True)

            # Apply filtering ON THE DATABASE SIDE if filtering is enabled
            if validatable_only:
                valid_activities = list(VALIDATABLE_ACTIVITIES.keys())
                # Use 'in_' operator for list matching
                query = query.in_('activity_name', valid_activities)

            # Apply pagination on the database side
            # Ensure offset and limit are non-negative
            offset = max(0, offset)
            limit = max(1, limit) # Ensure limit is at least 1
            query = query.range(offset, offset + limit - 1)

            result = query.execute()

            # The result object now contains 'data' and 'count'
            messages = result.data if result.data else [] # Ensure messages is a list even if data is None
            total_count = result.count if result.count is not None else 0 # Ensure count is an integer

            # Enrich messages (no need for Python filtering if done in DB)
            enriched_messages = []
            for msg in messages:
                    activity = msg.get('activity_name', '')
                    is_valid = is_activity_validatable(activity) # Still useful to add flag
                    msg['is_validatable'] = is_valid
                    msg['validation_info'] = get_activity_validation_info(activity)
                    enriched_messages.append(msg)

            # Return the list and total count for pagination purposes
            return {
                    'messages': enriched_messages,
                    'total_count': total_count, # Total matching VALIDATABLE messages in DB
                    'limit': limit,
                    'offset': offset
                }, 200

        except Exception as e:
            traceback.print_exc()
            # Ensure a dictionary is returned on error to match expected structure
            return {'error': str(e), 'messages': [], 'total_count': 0, 'limit': 0, 'offset': 0}, 500

# --- VALIDATION ENDPOINT ---

# === NEW, MERGED VALIDATE METHOD ===
@ns_validation.route('/validate/<string:message_id>')
class ValidateStoredMessage(Resource):
    @ns_validation.doc('validate_stored_message_with_storage')
    @ns_validation.expect(validation_with_stored_rules_parser)
    @ns_validation.marshal_with(lightweight_validation_response_model, code=200)
    def post(self, message_id):
        """Validate stored XML message using business rules from Supabase Storage"""
        if not supabase_client: 
            api.abort(500, "Supabase not configured.")
        
        args = validation_with_stored_rules_parser.parse_args()
        rule_id = args.get('rule_id')
        
        # Get the message
        result = supabase_client.table('xml_messages').select('*').eq('id', message_id).execute()
        if not result.data: 
            api.abort(404, f"Message with ID {message_id} not found.")
        
        message_record = result.data[0]
        xml_content = message_record['xml_content']
        activity_name = message_record.get('activity_name', 'UNKNOWN')

        # === NEW: Double-check activity is validatable ===
        if not is_activity_validatable(activity_name):
            api.abort(400, f'Activity "{activity_name}" cannot be validated - no business rules available')
        
        # Get the rule to use
        if rule_id:
            # Use specific rule requested
            rule_result = supabase_client.table('business_rules').select('*').eq('id', rule_id).execute()
            if not rule_result.data:
                api.abort(404, f"Business rule with ID {rule_id} not found")
            rule_used = rule_result.data[0]
        else:
            # Fall back to active rule
            active_rules = supabase_client.table('business_rules').select('*').eq('is_active', True).limit(1).execute()
            if not active_rules.data:
                api.abort(404, "No active business rules found. Please upload and activate rules first.")
            rule_used = active_rules.data[0]
        
        try:
            # Load the rule from storage
            rules_vectordb = load_business_rules_from_storage(rule_used['id'])
            
            # Create complete rule_info with all metadata
            rule_info = {
                'id': rule_used['id'],
                'name': rule_used['name'],
                'version': rule_used['version'],
                'file_name': rule_used['file_name'],
                'file_url': rule_used['file_url']
            }
            
            # Perform validation
            validation_response = validate_xml_against_rules(xml_content, rules_vectordb, rule_info)
            
            # Update message with validation result AND save which rule was used
            update_data = {
                'last_validation_result': validation_response,
                'validation_count': message_record.get('validation_count', 0) + 1,
                'business_rule_id': rule_used['id']
            }
            supabase_client.table('xml_messages').update(update_data).eq('id', message_id).execute()

            # Save to validation history with rule tracking
            print("Saving validation event to validation_history table...")
            history_data = {
                'message_id': message_id,
                'decision': validation_response['decision'],
                'explanation': validation_response['summary_explanation'],
                'technical_reasons': json.dumps(validation_response['validation_details']),
                'business_rules_used': f"{rule_used['name']} v{rule_used['version']}",
                'business_rule_id': rule_used['id']
            }
            supabase_client.table('validation_history').insert(history_data).execute()
            
            return validation_response, 200
            
        except Exception as e:
            traceback.print_exc()
            # Return a valid model on error
            return {
                'success': False,
                'decision': 'ERROR',
                'summary_explanation': f'Validation error: {str(e)}',
                'validation_details': {},
                'rule_used': None
            }, 500

# === THIS IS ONE OF THE MISSING ENDPOINTS ===
@ns_validation.route('/messages/<string:message_id>')
class MessageDetails(Resource):
    @ns_validation.doc('get_message_details')
    def get(self, message_id):
        """Fetches the details of a specific message by its ID."""
        if not supabase_client:
            return {'error': 'Supabase not configured'}, 500
        
        try:
            # Query the database for the message with the matching ID
            result = supabase_client.table('xml_messages').select('*').eq('id', message_id).single().execute()
            
            # Supabase's single() method handles the not-found case
            if not result.data:
                return {'error': 'Message not found'}, 404
                
            # Return raw data
            return result.data, 200
        except Exception as e:
            if "PGRST116" in str(e): # PostgREST error for "exact one row"
                 return {'error': f'Message with id {message_id} not found'}, 404
            return {'error': str(e)}, 500
            
# === THIS IS ONE OF THE MISSING ENDPOINTS ===
@ns_validation.route('/history/<string:message_id>')
class ValidationHistory(Resource):
    @ns_validation.doc('get_validation_history')
    def get(self, message_id):
        """Get validation history/audit trail for a specific message"""
        if not supabase_client:
            return {'error': 'Supabase not configured'}, 500
        
        try:
            # First verify the message exists
            message_result = supabase_client.table('xml_messages').select('id').eq('id', message_id).limit(1).execute()
            if not message_result.data:
                return {'error': 'Message not found'}, 404
            
            # Get all validation history for this message
            history_result = supabase_client.table('validation_history').select('*').eq('message_id', message_id).order('created_at', desc=True).execute()
            
            return {
                'success': True,
                'message_id': message_id,
                'validation_count': len(history_result.data),
                'history': history_result.data
            }, 200
            
        except Exception as e:
            return {'error': str(e)}, 500
            
# --- HUMAN VERIFICATION ENDPOINTS (from old file) ---
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

# === NEW: ACTIVITIES NAMESPACE ===
@ns_activities.route('/validatable')
class ValidatableActivities(Resource):
    @ns_activities.doc('list_validatable_activities')
    def get(self):
        """List all activities that can be validated"""
        return {
            'success': True,
            'count': len(VALIDATABLE_ACTIVITIES),
            'activities': [
                {
                    'name': name,
                    **info
                }
                for name, info in VALIDATABLE_ACTIVITIES.items()
            ]
        }, 200

@ns_activities.route('/check/<string:activity_name>')
class CheckActivity(Resource):
    @ns_activities.doc('check_activity_validatable')
    def get(self, activity_name):
        """Check if specific activity is validatable"""
        is_valid = is_activity_validatable(activity_name)
        info = get_activity_validation_info(activity_name)
        
        return {
            'activity_name': activity_name,
            'is_validatable': is_valid,
            'validation_info': info
        }, 200

@ns_activities.route('/coverage')
class ActivityCoverage(Resource):
    @ns_activities.doc('get_activity_coverage')
    def get(self):
        """Get statistics on validation coverage"""
        if not supabase_client:
            return {'error': 'Supabase not configured'}, 500
        
        try:
            result = supabase_client.table('xml_messages').select('activity_name').execute()
            
            total = len(result.data)
            validatable_count = 0
            activity_breakdown = {}
            
            for msg in result.data:
                activity = msg.get('activity_name', 'UNKNOWN')
                is_valid = is_activity_validatable(activity)
                
                if is_valid:
                    validatable_count += 1
                
                if activity not in activity_breakdown:
                    activity_breakdown[activity] = {
                        'count': 0,
                        'is_validatable': is_valid
                    }
                activity_breakdown[activity]['count'] += 1
            
            return {
                'success': True,
                'total_messages': total,
                'validatable_count': validatable_count,
                'coverage_percentage': round((validatable_count / total * 100), 2) if total > 0 else 0,
                'activity_breakdown': [
                    {'activity': k, **v}
                    for k, v in sorted(activity_breakdown.items(), key=lambda x: x[1]['count'], reverse=True)
                ]
            }, 200
        except Exception as e:
            return {'error': str(e)}, 500
            
# --- ERROR HANDLERS (from old file) ---
@app.errorhandler(404)
def not_found_error(error):
    # Check if it was a flask-restx 404 or a general 404
    if hasattr(error, 'data'):
        return jsonify(error.data), 404
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    if hasattr(error, 'data'):
        return jsonify(error.data), 500
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# Initialize storage bucket on startup
def initialize_storage():
    if create_storage_bucket_if_not_exists():
        print(f"Storage bucket '{STORAGE_BUCKET}' is ready")
    else:
        print(f"Warning: Could not initialize storage bucket '{STORAGE_BUCKET}'")

# --- MAIN APP EXECUTION ---
if __name__ == '__main__':
    print("="*80)
    print("🚀 Starting XML Validation API v2.1 (Merged Version)")
    print("="*80)
    print(f"\n📊 Validation Coverage:")
    print(f"   - {len(VALIDATABLE_ACTIVITIES)} activities have business rules")
    print(f"   - Only validatable meldingen accepted via POST /validation/messages\n")
    print("🌐 Swagger UI: http://localhost:5000/docs/\n")
    print("📁 Activity Endpoints:")
    print("   - GET  /api/v1/activities/validatable")
    print("   - GET  /api/v1/activities/check/<name>")
    print("   - GET  /api/v1/activities/coverage")
    print("\n✅ Validation Endpoints (Filtered):")
    print("   - POST /api/v1/validation/messages      (ONLY validatable)")
    print("   - GET  /api/v1/validation/messages      (filtered by default)")
    print("   - GET  /api/v1/validation/messages/{id} (FIXED)")
    print("   - POST /api/v1/validation/validate/{id}")
    print("   - GET  /api/v1/validation/history/{id}  (FIXED)")
    print("\nRULES Endpoints:")
    print("   - GET  /api/v1/rules/list             (FIXED)")
    print("="*80)
    
    # Initialize storage on startup
    initialize_storage()
    
    app.run(host='0.0.0.0', port=5000, debug=True)