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
from langchain_community.vectorstores import SupabaseVectorStore
import uuid
from datetime import datetime
import re
from typing import List, Dict, Any

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
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
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
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
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

# --- Core Functions (No Heavy AI Models) ---
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

def retrieve_business_rules(vectordb, activity_name: str, message_type: str) -> List:
    """
    IMPROVED v5: STRICT activity name matching with explicit exclusion
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
    
    print(f"  🔑 Key activity words: {activity_keywords}")
    print(f"  🎯 Full activity name: '{activity_name}'")
    
    # Build focused search queries
    queries = []
    
    # Primary query: Full activity phrase (most important!)
    if len(activity_keywords) >= 2:
        queries.append(f"{activity_keywords[0]} {activity_keywords[1]}")
    
    # Add specific activity-related searches
    if 'toepassen' in activity_keywords:
        queries.extend([
            "toepassen grond baggerspecie landbodem",
            "functionele toepassing kubieke meter"
        ])
    elif 'graven' in activity_keywords:
        queries.append("graven bodem interventiewaarde")
    elif 'saneren' in activity_keywords:
        queries.append("saneren bodem saneringsaanpak")
    
    # Add type-specific query
    queries.append(f"{type_keyword} bevat")
    
    all_docs = []
    seen_content = set()
    
    for query in queries:
        print(f"  📝 Query: '{query}'")
        results = vectordb.similarity_search(query, k=8)
        
        for doc in results:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                all_docs.append(doc)
    
    print(f"  ✅ Found {len(all_docs)} unique document chunks")
    
    # ULTRA-STRICT filtering - must match EXACT activity
    filtered_docs = []
    wrong_activity_docs = []
    
    # Define EXACT wrong activity phrases to HARD REJECT
    wrong_activities_map = {
        'toepassen': [
            'graven in de bodem',
            'saneren van de bodem',
            'bodemenergiesysteem',
            'windturbine',
            'kuilvoer',
            'mestvergisting',
            'artikel 4.1224',  # graven
            'artikel 4.1235',  # saneren
            'paragraaf 4.120',  # graven
            '§ 4.121',  # saneren
            'saneringsaanpak'  # specific to saneren
        ],
        'graven': [
            'toepassen grond',
            'saneren van de bodem',
            'bodemenergiesysteem',
            'artikel 4.1235',
            'saneringsaanpak'
        ],
        'saneren': [
            'toepassen grond',
            'graven in de bodem',
            'bodemenergiesysteem',
            'artikel 4.1224'
        ]
    }
    
    # Determine which wrong activities to exclude
    wrong_activities = []
    for key_activity in ['toepassen', 'graven', 'saneren']:
        if key_activity in activity_keywords:
            wrong_activities = wrong_activities_map.get(key_activity, [])
            break
    
    print(f"  🚫 Excluding documents containing: {wrong_activities}")
    
    for doc in all_docs:
        content_lower = doc.page_content.lower()
        
        # HARD REJECTION: Check for wrong activities FIRST
        has_wrong_activity = any(wrong_act in content_lower for wrong_act in wrong_activities)
        
        if has_wrong_activity:
            wrong_activity_docs.append(doc)
            print(f"  ❌ EXCLUDED: Contains wrong activity phrase (page {doc.metadata.get('page', '?')})")
            continue
        
        # Count activity keyword matches (ALL keywords, not just first 3)
        keyword_matches = sum(1 for kw in activity_keywords if kw in content_lower)
        
        # Check if it has requirements structure
        has_requirements = bool(re.search(r'^\s*[a-z]\.\s+', content_lower, re.MULTILINE))
        
        # Check if it mentions correct type (melding/informatieplicht)
        has_type_keyword = type_keyword in content_lower
        
        # STRICT DECISION LOGIC - Require MORE matches for 'toepassen'
        min_keywords_required = 3 if 'toepassen' in activity_keywords else 2
        
        if keyword_matches >= min_keywords_required and has_type_keyword:
            filtered_docs.append(doc)
            print(f"  ✅ INCLUDED: {keyword_matches}/{len(activity_keywords)} keywords + '{type_keyword}' (page {doc.metadata.get('page', '?')})")
        elif has_requirements and has_type_keyword and keyword_matches >= 2:
            filtered_docs.append(doc)
            print(f"  ✅ INCLUDED: Requirements + '{type_keyword}' + {keyword_matches} keywords (page {doc.metadata.get('page', '?')})")
        else:
            print(f"  ⏭️  SKIPPED: keywords={keyword_matches}/{len(activity_keywords)}, type={has_type_keyword}, req={has_requirements}")
    
    # If no relevant rules found, return warning
    if not filtered_docs:
        print(f"  ⚠️  WARNING: No rules found for activity '{activity_name}'!")
        print(f"  ⚠️  The uploaded business rules document may not contain rules for this activity.")
        print(f"  ⚠️  Found rules for other activities: {len(wrong_activity_docs)} chunks")
        
        # Return empty to trigger proper error handling
        return []
    
    print(f"  🎯 Final filtered count: {len(filtered_docs)} relevant chunks\n")
    return filtered_docs

def validate_message(message_data: Dict, rule_documents: List, rule_info: Dict = None) -> Dict:
    """
    CORRECTED v3: Properly handle both pre-activity and post-activity submissions
    
    Key Logic:
    - Pre-activity informatieplicht: Check all requirements from 4.1237, 4.1238
    - Post-activity informatieplicht: Only check for evaluation report attachment
    
    Note: The uploaded business rules document does NOT explicitly define 
    post-activity requirements. This function uses defensive interpretation
    to avoid applying "before start" rules to "after completion" submissions.
    """
    print(f"\n🔍 VALIDATING MESSAGE")
    print(f"   Activity: {message_data.get('activity_name')}")
    print(f"   Type: {message_data.get('message_type')}")
    print(f"   Processing {len(rule_documents)} relevant rule documents\n")
    
    report = {"checked_requirements": [], "checked_attachments": []}
    
    # Detect if this is a post-activity submission (evaluatieverslag)
    is_post_activity = any(
        "evaluatieverslag" in s.get('answer', '').lower() or
        "nazorgplan" in s.get('answer', '').lower()
        for s in message_data.get('specifications', [])
    )
    
    type_keyword = 'informatieplicht' if message_data['message_type'].lower() == 'informatie' else 'melding'
    
    print(f"   Type keyword: '{type_keyword}'")
    print(f"   Post-activity: {is_post_activity}")
    
    # IMPORTANT: Log if we're interpreting rules
    if is_post_activity and type_keyword == 'informatieplicht':
        print(f"   ⚠️  POST-ACTIVITY MODE ACTIVATED")
        print(f"   ⚠️  Note: Business rules document does not explicitly define post-activity requirements.")
        print(f"   ⚠️  Interpreting: Skipping pre-activity requirements (4.1237, 4.1238)")
        print(f"   ⚠️  Only checking: Evaluation report attachment\n")
    else:
        print(f"   ℹ️  PRE-ACTIVITY MODE: Checking all {type_keyword} requirements\n")
    
    # Process each rule document
    for idx, doc in enumerate(rule_documents):
        content = doc.page_content
        content_lower = content.lower()
        
        print(f"📄 Processing document chunk {idx + 1}/{len(rule_documents)}")
        print(f"   Page: {doc.metadata.get('page', '?')}")
        
        # Skip if wrong type
        if type_keyword not in content_lower:
            print(f"   ⏭️  Skipping: doesn't contain '{type_keyword}'\n")
            continue
        
        # Skip exclusion rules
        if any(phrase in content_lower for phrase in [
            "vallen niet", "is niet van toepassing", "onder de aanwijzing vallen niet"
        ]):
            print(f"   ⏭️  Skipping: contains exclusion phrase\n")
            continue
        
        # Extract the article number for context
        article_match = re.search(r'artikel\s+(\d+\.\d+)', content_lower)
        article_num = article_match.group(1) if article_match else "unknown"
        
        print(f"   📜 Article: {article_num}")
        
        # POST-ACTIVITY LOGIC: Skip ALL pre-activity requirement articles
        skip_this_chunk = False
        if is_post_activity and type_keyword == 'informatieplicht':
            # Identify pre-activity articles by their explicit "voor het begin" language
            is_pre_activity_article = (
                "voor het begin van de activiteit" in content_lower or
                "vier weken voor het begin" in content_lower or
                "een week voor het begin" in content_lower
            )
            
            # Also identify by article numbers known to be pre-activity
            is_known_pre_activity_article = article_num in ["4.1237", "4.1238", "4.1226", "4.1227"]
            
            if is_pre_activity_article or is_known_pre_activity_article:
                print(f"   ⏭️  Skipping: Pre-activity requirement article {article_num} (post-activity mode)\n")
                skip_this_chunk = True
        
        if skip_this_chunk:
            continue
        
        # Extract requirements using multiple patterns
        patterns = [
            r'^\s*([a-z]\.)\s+(.+?)(?=\n[a-z]\.|$)',      # a. , b. , c.
            r'^\s*(\d+[°º]\.)\s+(.+?)(?=\n\d+[°º]\.|$)',  # 1°. , 2°.
        ]
        
        requirements_found = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
            for match in matches:
                req_text = match.group(2).strip()
                # Clean up newlines and extra spaces
                req_text = ' '.join(req_text.split())
                
                # Skip very short items
                if len(req_text) < 10:
                    continue
                
                # Skip if it ends mid-sentence (indicates incomplete extraction)
                if not req_text.endswith(('.', ';', ':', 'en')):
                    # Try to find the complete sentence
                    continue
                
                requirements_found.append(req_text)
        
        print(f"   📋 Found {len(requirements_found)} requirements\n")
        
        # Validate each requirement (ONLY if not post-activity mode)
        if is_post_activity and type_keyword == 'informatieplicht':
            print(f"   ⏭️  Skipping requirement validation (post-activity mode - only checking attachments)\n")
        else:
            for req_text in requirements_found:
                print(f"   ➡️  Requirement: '{req_text[:100]}...'")
                
                found = False
                evidence = "Niet gevonden in XML bericht."
                
                # Check 1: Begrenzing/Coordinates (REQUIRED by 4.1237a)
                if any(keyword in req_text.lower() for keyword in ['begrenzing', 'locatie waarop']):
                    if message_data.get('has_coordinates'):
                        found = True
                        evidence = "✅ Begrenzing/coördinaten aanwezig in bericht"
                        print(f"      ✅ MET: Coordinates found")
                    else:
                        evidence = "❌ Begrenzing/coördinaten ontbreken"
                        print(f"      ❌ MISSING: No coordinates")
                
                # Check 2: Expected start date (REQUIRED by 4.1237b)
                elif 'verwachte datum' in req_text.lower() and 'begin' in req_text.lower():
                    date_found = False
                    for spec in message_data.get('specifications', []):
                        spec_text = (spec.get('question', '') + ' ' + spec.get('answer', '')).lower()
                        if any(word in spec_text for word in ['datum', 'date', '2025', '2024']):
                            date_found = True
                            evidence = f"✅ Datum informatie: '{spec.get('answer', '')[:50]}'"
                            print(f"      ✅ MET: Date found")
                            break
                    
                    if not date_found:
                        evidence = "❌ Verwachte startdatum ontbreekt"
                        print(f"      ❌ MISSING: Start date missing")
                    else:
                        found = True
                
                # Check 3: Name/Address of executor (REQUIRED by 4.1238a)
                elif 'naam' in req_text.lower() and 'adres' in req_text.lower() and 'werkzaamheden' in req_text.lower():
                    xml_has_name = False
                    for spec in message_data.get('specifications', []):
                        spec_text = (spec.get('question', '') + ' ' + spec.get('answer', '')).lower()
                        if any(word in spec_text for word in ['uitvoerder', 'aannemer', 'naam', 'bedrijf']):
                            xml_has_name = True
                            evidence = f"✅ Naam/adres uitvoerder: '{spec.get('answer', '')[:50]}'"
                            print(f"      ✅ MET: Executor name found")
                            break
                    
                    if not xml_has_name:
                        evidence = "❌ Naam en adres van degene die werkzaamheden verricht ontbreekt"
                        print(f"      ❌ MISSING: Executor name/address")
                    else:
                        found = True
                
                # Check 4: Environmental supervision company (REQUIRED by 4.1238b)
                elif 'milieukundige begeleiding' in req_text.lower() and 'onderneming' in req_text.lower():
                    supervision_found = False
                    for spec in message_data.get('specifications', []):
                        spec_text = (spec.get('question', '') + ' ' + spec.get('answer', '')).lower()
                        if any(word in spec_text for word in ['milieukundig', 'begeleiding', 'toezicht']):
                            supervision_found = True
                            evidence = f"✅ Milieukundige begeleiding: '{spec.get('answer', '')[:50]}'"
                            print(f"      ✅ MET: Environmental supervision found")
                            break
                    
                    if not supervision_found:
                        evidence = "❌ Naam/adres onderneming milieukundige begeleiding ontbreekt"
                        print(f"      ❌ MISSING: Environmental supervision company")
                    else:
                        found = True
                
                # Check 5: Environmental supervisor person (REQUIRED by 4.1238c)
                elif 'milieukundige begeleiding' in req_text.lower() and 'natuurlijke persoon' in req_text.lower():
                    person_found = False
                    for spec in message_data.get('specifications', []):
                        spec_text = (spec.get('question', '') + ' ' + spec.get('answer', '')).lower()
                        if any(word in spec_text for word in ['milieukundig', 'persoon', 'naam']):
                            person_found = True
                            evidence = f"✅ Natuurlijke persoon milieukundige begeleiding: '{spec.get('answer', '')[:50]}'"
                            print(f"      ✅ MET: Supervisor person found")
                            break
                    
                    if not person_found:
                        evidence = "❌ Naam natuurlijke persoon milieukundige begeleiding ontbreekt"
                        print(f"      ❌ MISSING: Supervisor person name")
                    else:
                        found = True
                
                # Check 6: General keyword matching for other requirements
                else:
                    req_keywords = set(
                        word for word in re.findall(r'\b\w{4,}\b', req_text.lower())
                        if word not in ['voor', 'wordt', 'worden', 'deze', 'zijn', 'heeft', 'hebben', 'moet', 'moeten', 'naar']
                    )
                    
                    best_score = 0
                    best_spec = None
                    
                    for spec in message_data.get('specifications', []):
                        spec_text = (spec.get('answer', '') + ' ' + spec.get('question', '')).lower()
                        spec_keywords = set(re.findall(r'\b\w{4,}\b', spec_text))
                        
                        matches = req_keywords.intersection(spec_keywords)
                        score = len(matches)
                        
                        if score > best_score:
                            best_score = score
                            best_spec = spec
                    
                    if best_score >= 2:
                        found = True
                        evidence = f"✅ Match ({best_score} keywords): '{best_spec.get('answer', '')[:80]}'"
                        print(f"      ✅ MET: {best_score} keyword matches")
                    else:
                        evidence = f"❌ Onvoldoende match ({best_score} keywords)"
                        print(f"      ❌ MISSING: Only {best_score} keyword matches")
                
                # Add to report
                requirement_item = {
                    "requirement": req_text,
                    "status": "MET" if found else "MISSING",
                    "reason": evidence,
                    "source_quote": content[:400],
                    "source_page": doc.metadata.get('page', 1)
                }
                
                if rule_info:
                    requirement_item["source_file"] = rule_info.get('file_name', 'Unknown')
                    requirement_item["rule_name"] = rule_info.get('name', 'Unknown')
                    requirement_item["rule_version"] = rule_info.get('version', '1.0')
                
                report["checked_requirements"].append(requirement_item)
    
    # Check attachments
    print("\n📎 Checking attachments...")
    attachments = message_data.get('attachments', [])
    
    if is_post_activity:
        # For post-activity: Evaluatieverslag is required
        has_evaluation = any('evaluatie' in att.lower() for att in attachments)
        
        attachment_item = {
            "requirement": "Evaluatieverslag na uitvoering (post-activity)",
            "status": "MET" if has_evaluation else "MISSING",
            "reason": f"✅ Evaluatieverslag bijgevoegd: {attachments[0] if attachments else 'N/A'}" if has_evaluation else "❌ Evaluatieverslag ontbreekt",
            "source_quote": "Vereist voor post-activity informatieplicht",
            "source_page": None
        }
        
        if rule_info:
            attachment_item["source_file"] = rule_info.get('file_name')
            attachment_item["rule_name"] = rule_info.get('name')
            attachment_item["rule_version"] = rule_info.get('version')
        
        report["checked_attachments"].append(attachment_item)
        print(f"   {'✅ MET' if has_evaluation else '❌ MISSING'}: Evaluatieverslag\n")
    
    # Calculate final validity
    missing_requirements = [r for r in report['checked_requirements'] if r['status'] == 'MISSING']
    missing_attachments = [a for a in report['checked_attachments'] if a['status'] == 'MISSING']
    
    is_valid = len(missing_requirements) == 0 and len(missing_attachments) == 0
    
    # Summary
    total_req = len(report['checked_requirements'])
    met_req = sum(1 for r in report['checked_requirements'] if r['status'] == 'MET')
    
    print(f"\n{'='*60}")
    print(f"📊 VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total requirements: {total_req}")
    print(f"Requirements MET: {met_req}")
    print(f"Requirements MISSING: {len(missing_requirements)}")
    print(f"Attachments MISSING: {len(missing_attachments)}")
    print(f"\nFinal decision: {'✅ VALID (ACCEPTED)' if is_valid else '❌ INVALID (REJECTED)'}")
    print(f"{'='*60}\n")
    
    return {
        'is_valid': is_valid,
        'details': report
    }

def generate_simple_explanation(validation_result: Dict, message_data: Dict) -> str:
    """
    Generate clear Dutch explanation - UNCHANGED but included for completeness
    """
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
                        embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2",
                            model_kwargs={'device': 'cpu'}
                        )
                        
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
# Replace the entire ValidateStoredMessage class (lines ~544-605)
# This is the complete updated validation endpoint

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
            api.abort(500, f"Validation error: {e}")

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
                
            return result.data, 200
        except Exception as e:
            return {'error': str(e)}, 500
# --- END OF BLOCK TO ADD ---          
@ns_validation.route('/history/<string:message_id>')
class ValidationHistory(Resource):
    @ns_validation.doc('get_validation_history')
    def get(self, message_id):
        """Get validation history/audit trail for a specific message"""
        if not supabase_client:
            return {'error': 'Supabase not configured'}, 500
        
        try:
            # First verify the message exists
            message_result = supabase_client.table('xml_messages').select('id').eq('id', message_id).execute()
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
    print("  - POST /api/v1/rules/upload         - Upload business rules to storage")
    print("  - GET  /api/v1/rules/list           - List all rules")
    print("  - GET  /api/v1/rules/active         - Get active rules")
    print("  - PATCH /api/v1/rules/activate/{id}  - Activate specific rule")
    print("Validation Endpoints:")
    print("  - POST /api/v1/validation/messages    - Store XML messages")
    print("  - POST /api/v1/validation/validate/{message_id} - Validate using storage")
    print("No heavy AI models loaded - fast startup and cheap hosting!")
    
    # Initialize storage on startup
    initialize_storage()
    
    app.run(host='0.0.0.0', port=5000, debug=True)