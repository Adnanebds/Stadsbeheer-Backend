# Google Colab Implementation with Gemma 2-2b for Response Generation

# Step 1: Install required libraries
!pip install langchain langchain-community faiss-cpu huggingface_hub transformers sentence-transformers pypdf python-docx unstructured

# Step 2: Import libraries with correct packages
import os
import re
import xml.etree.ElementTree as ET
# Updated imports for LangChain
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredFileLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import json

# Step 3: Upload files
# In Colab, run this code cell to show file upload widgets
from google.colab import files

print("Upload the business rules document (any format - PDF, DOCX, or TXT):")
business_rules_file = files.upload()  # This will prompt you to upload the business rules file
business_rules_filename = list(business_rules_file.keys())[0]

print("Upload the sample XML message file:")
xml_file = files.upload()  # This will prompt you to upload the XML file
xml_filename = list(xml_file.keys())[0]

# Step 4: Set up Hugging Face (if you have a token)
# If you don't have a token, you can still use the models but with download limitations
HF_TOKEN = ""  # Enter your token here if you have one
if HF_TOKEN:
    login(HF_TOKEN)

# Step 5: Set up Gemma 2-2b model for response generation
print("Setting up Gemma 2-2b model for response generation...")
model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create text generation pipeline
generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.1,  # Lower temperature for more precise responses
    device=0 if os.environ.get("CUDA_VISIBLE_DEVICES") else -1  # Use GPU if available
)

# Improved file loader that handles different file types and encodings
def load_business_rules(file_path):
    # Determine file type and use appropriate loader
    file_extension = file_path.split('.')[-1].lower()
    
    try:
        if file_extension == 'pdf':
            loader = PyPDFLoader(file_path)  # Changed from PDFLoader to PyPDFLoader
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
            print(chunks[0].page_content[:500] + "..." if len(chunks[0].page_content) > 500 else chunks[0].page_content)
        
        # Initialize vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(chunks, embeddings)
        
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

# Fixed XML file reading function that handles different encodings
def read_xml_file(file_path):
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
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

# Improved XML parser that detects message type
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
        
        return {
            'activity_name': activity_name,
            'message_type': message_type,
            'specifications': specifications,
            'attachments': attachments,
            'has_coordinates': has_coordinates
        }
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return None

# Improved business rule retrieval that considers message type
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
        print(f"Searching for: {query}")
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
    
    return rule_texts

# More sophisticated rule extraction
def extract_requirements(rule_texts, message_type):
    requirements = {
        'required_fields': [],
        'required_attachments': []
    }
    
    # Look for specific patterns based on message type
    message_type_lower = message_type.lower() if message_type else ""
    
    for text in rule_texts:
        print(f"\nAnalyzing text chunk for requirements ({len(text)} characters):")
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
                    print(f"  Found required field: {clean_field}")
                    requirements['required_fields'].append(clean_field)
            
            # Check for attachment requirements
            if "evaluatieverslag" in text.lower():
                print("  Found attachment requirement: evaluatieverslag")
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
                    print(f"  Found required field: {clean_field}")
                    requirements['required_fields'].append(clean_field)
            
            # Check for attachment requirements in melding
            if "evaluatieverslag" in text.lower():
                print("  Found attachment requirement: evaluatieverslag")
                requirements['required_attachments'].append("evaluatieverslag")
    
    return requirements

# Enhanced validation function that handles coordinates as boundary info
def validate_message(message_data, requirements):
    print("\nValidating message against requirements:")
    
    # Check if all required fields are present
    missing_fields = []
    for field in requirements['required_fields']:
        field_found = False
        
        # Special case for boundary location information
        if "begrenzing van de locatie" in field.lower() and message_data.get('has_coordinates', False):
            field_found = True
            print(f"  ✓ Required field found: '{field}' (via coordinates in message)")
            continue
        
        # Check specifications
        for spec in message_data['specifications']:
            if (spec['answer'] and field.lower() in spec['answer'].lower()) or (
                spec['question'] and field.lower() in spec['question'].lower()
            ):
                field_found = True
                print(f"  ✓ Required field found: '{field}'")
                break
        
        if not field_found:
            print(f"  ✗ Required field missing: '{field}'")
            missing_fields.append(field)
    
    # Check if all required attachments are present
    missing_attachments = []
    for attachment in requirements['required_attachments']:
        if not message_data['attachments']:
            print(f"  ✗ Required attachment missing: '{attachment}' (no attachments present)")
            missing_attachments.append(attachment)
            continue
        
        attachment_found = False
        for att in message_data['attachments']:
            if attachment.lower() in att.lower():
                attachment_found = True
                print(f"  ✓ Required attachment found: '{attachment}' in '{att}'")
                break
        
        if not attachment_found:
            print(f"  ✗ Required attachment missing: '{attachment}'")
            missing_attachments.append(attachment)
    
    # Determine if message is valid
    is_valid = len(missing_fields) == 0 and len(missing_attachments) == 0
    
    return {
        'is_valid': is_valid,
        'missing_fields': missing_fields,
        'missing_attachments': missing_attachments
    }

# NEW FUNCTION: Generate explanation using Gemma 2-2b
def generate_explanation(validation_result, message_data):
    activity = message_data.get('activity_name', 'Unknown activity')
    message_type = message_data.get('message_type', 'Unknown type')
    
    if validation_result['is_valid']:
        prompt = f"""
        Generate a clear, concise explanation for why a message was accepted.
        
        Activity: {activity}
        Message Type: {message_type}
        
        The message contains all required fields and attachments according to the business rules.
        
        Response:
        """
    else:
        missing_fields = ", ".join(validation_result['missing_fields'])
        missing_attachments = ", ".join(validation_result['missing_attachments']) if validation_result['missing_attachments'] else "None"
        
        prompt = f"""
        Generate a clear, concise explanation for why a message was rejected.
        
        Activity: {activity}
        Message Type: {message_type}
        Missing Fields: {missing_fields}
        Missing Attachments: {missing_attachments}
        
        The message does not meet all requirements according to the business rules. Explain what needs to be added.
        
        Response:
        """
    
    # Generate response using Gemma 2-2b
    try:
        result = generation_pipeline(prompt, max_length=300)[0]['generated_text']
        
        # Extract just the generated response part
        explanation = result.split('Response:')[-1].strip()
        return explanation
    except Exception as e:
        print(f"Error generating explanation: {e}")
        # Fallback to simple explanation
        if validation_result['is_valid']:
            return "The message meets all requirements according to the business rules."
        else:
            return f"The message is missing required information: {missing_fields}"

# Main validation function with Gemma explanation
def validate_xml_against_rules(xml_content, vectordb):
    print("\n========== STARTING VALIDATION ==========\n")
    
    # 1. Parse the message
    print("Step 1: Parsing XML message")
    message_data = parse_xml_message(xml_content)
    if not message_data:
        return {"decision": "REJECTED", "explanation": "Failed to parse XML"}
    
    print(f"\nParsed message data:")
    print(f"  Activity: {message_data['activity_name']}")
    print(f"  Message Type: {message_data['message_type']}")
    print(f"  Specifications: {len(message_data['specifications'])} items")
    print(f"  Attachments: {message_data['attachments']}")
    
    # 2. Retrieve relevant business rules
    print("\nStep 2: Retrieving relevant business rules")
    rule_texts = retrieve_business_rules(
        vectordb, 
        message_data['activity_name'], 
        message_data['message_type']
    )
    
    print(f"\nRetrieved {len(rule_texts)} rule text chunks")
    
    # 3. Extract structured requirements
    print("\nStep 3: Extracting structured requirements")
    requirements = extract_requirements(rule_texts, message_data['message_type'])
    
    print(f"\nExtracted requirements summary:")
    print(f"  Required fields: {len(requirements['required_fields'])}")
    print(f"  Required attachments: {len(requirements['required_attachments'])}")
    
    # 4. Validate the message
    print("\nStep 4: Validating message")
    validation_result = validate_message(message_data, requirements)
    
    # 5. Generate explanation with Gemma 2-2b
    print("\nStep 5: Generating explanation with Gemma 2-2b")
    explanation = generate_explanation(validation_result, message_data)
    
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
    
    print("\n========== VALIDATION COMPLETE ==========")
    return result

# Step 6: Run the validation
print("\nInitializing vector database with business rules...")
rules_vectordb = load_business_rules(business_rules_filename)

print("\nReading XML message file...")
xml_content = read_xml_file(xml_filename)

print("\nRunning validation...")
result = validate_xml_against_rules(xml_content, rules_vectordb)

print("\nValidation Result:")
print(f"Decision: {result['decision']}")
print(f"Technical Reasons: {result['technical_reasons']}")
print(f"Explanation (Gemma 2-2b): {result['explanation']}")