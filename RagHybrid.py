# Local Development Version - Compatible with Visual Studio

# Step 1: First install dependencies (run this in your terminal/command prompt)
# pip install langchain langchain-community faiss-cpu huggingface_hub transformers sentence-transformers pypdf python-docx unstructured torch

# Step 2: Import libraries
import os
import re
import xml.etree.ElementTree as ET
import argparse
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredFileLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import json

# Step 3: Setup command line arguments (for local execution)
def setup_argparse():
    parser = argparse.ArgumentParser(description='Validate XML messages against business rules')
    parser.add_argument('--rules', type=str, help='Path to business rules document file')
    parser.add_argument('--xml', type=str, help='Path to XML message file') 
    parser.add_argument('--output', type=str, default='validation_result.json', help='Path to save validation result')
    parser.add_argument('--rules_db', type=str, default='business_rules_db.json', help='Path to save extracted rules')
    parser.add_argument('--interactive', action='store_true', help='Use interactive file selection mode')
    return parser.parse_args()

# Interactive file selection for when no paths are provided
def get_file_path_interactive(file_type="file"):
    """
    Interactive file path selection - allows user to browse or enter path
    """
    import tkinter as tk
    from tkinter import filedialog
    
    try:
        # Try to use GUI file dialog
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        if file_type == "rules":
            file_path = filedialog.askopenfilename(
                title="Select Business Rules Document",
                filetypes=[
                    ("All supported", "*.pdf *.docx *.doc *.txt"),
                    ("PDF files", "*.pdf"),
                    ("Word documents", "*.docx *.doc"),
                    ("Text files", "*.txt"),
                    ("All files", "*.*")
                ]
            )
        elif file_type == "xml":
            file_path = filedialog.askopenfilename(
                title="Select XML Message File",
                filetypes=[
                    ("XML files", "*.xml"),
                    ("All files", "*.*")
                ]
            )
        else:
            file_path = filedialog.askopenfilename(title=f"Select {file_type}")
            
        root.destroy()
        return file_path if file_path else None
        
    except ImportError:
        # Fallback to command line input if tkinter is not available
        print(f"\nPlease enter the path to your {file_type} file:")
        print("(You can drag and drop the file here, or type the full path)")
        
        while True:
            file_path = input(f"{file_type.capitalize()} file path: ").strip()
            
            # Remove quotes if user copied path with quotes
            file_path = file_path.strip('"').strip("'")
            
            if os.path.exists(file_path):
                return file_path
            else:
                print(f"File not found: {file_path}")
                retry = input("Try again? (y/n): ").lower()
                if retry != 'y':
                    return None

# Dynamic file path resolution
def resolve_file_paths(args):
    """
    Resolve file paths - either from arguments, interactive selection, or default locations
    """
    rules_path = None
    xml_path = None
    
    # Try to get rules file path
    if args.rules:
        if os.path.exists(args.rules):
            rules_path = args.rules
        else:
            print(f"Rules file not found: {args.rules}")
    
    # Try to get XML file path  
    if args.xml:
        if os.path.exists(args.xml):
            xml_path = args.xml
        else:
            print(f"XML file not found: {args.xml}")
    
    # If interactive mode or files not found, prompt user
    if args.interactive or not rules_path:
        print("\n=== Business Rules Document Selection ===")
        rules_path = get_file_path_interactive("rules")
        if not rules_path:
            print("No business rules file selected. Exiting.")
            return None, None
    
    if args.interactive or not xml_path:
        print("\n=== XML Message File Selection ===")
        xml_path = get_file_path_interactive("xml")
        if not xml_path:
            print("No XML message file selected. Exiting.")
            return None, None
    
    # Final check if we still don't have paths - look in current directory
    if not rules_path:
        # Look for common rule file names in current directory
        possible_rules = [f for f in os.listdir('.') if f.lower().endswith(('.pdf', '.docx', '.doc', '.txt')) and 'rule' in f.lower()]
        if possible_rules:
            print(f"\nFound possible rules file: {possible_rules[0]}")
            use_file = input(f"Use '{possible_rules[0]}'? (y/n): ").lower()
            if use_file == 'y':
                rules_path = possible_rules[0]
    
    if not xml_path:
        # Look for XML files in current directory
        possible_xml = [f for f in os.listdir('.') if f.lower().endswith('.xml')]
        if possible_xml:
            print(f"\nFound possible XML file: {possible_xml[0]}")
            use_file = input(f"Use '{possible_xml[0]}'? (y/n): ").lower()
            if use_file == 'y':
                xml_path = possible_xml[0]
    
    return rules_path, xml_path

# Optional: Hugging Face login if you have a token
def setup_huggingface(token=None):
    if token:
        login(token)
    print("Setting up models...")

# Setup Gemma 2-2b model
def setup_gemma_model():
    try:
        model_name = "google/gemma-2-2b-it"
        print(f"Loading {model_name}...")
        
        # For CPU-only environments, add device_map="auto" and low_cpu_mem_usage=True
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto",  # This will use CPU if no GPU is available
            low_cpu_mem_usage=True
        )
        
        # Create pipeline with lower precision for better performance
        generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.1
        )
        return generation_pipeline
    except Exception as e:
        print(f"Error loading Gemma model: {e}")
        print("Falling back to smaller model for text generation...")
        # Fallback to a much smaller model if Gemma fails to load
        try:
            return pipeline("text-generation", model="distilgpt2", max_length=300)
        except:
            print("Could not load any text generation model. Explanations will be basic.")
            return None

# Save extracted rules to JSON file
def save_extracted_rules(activity_name, message_type, requirements, output_file="extracted_rules.json"):
    """
    Save the extracted requirements to a JSON file.
    """
    try:
        # Try to load existing file if it exists
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                rules_db = json.load(f)
        else:
            rules_db = {}
        
        # Add or update rules for this activity and message type
        if activity_name not in rules_db:
            rules_db[activity_name] = {}
        rules_db[activity_name][message_type.lower()] = {
            "required_fields": requirements["required_fields"],
            "required_attachments": requirements["required_attachments"]
        }
        
        # Save the updated rules
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(rules_db, f, indent=2, ensure_ascii=False)
        
        print(f"Rules saved to {output_file}")
        
    except Exception as e:
        print(f"Error saving rules to file: {e}")

# Load business rules document
def load_business_rules(file_path):
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
        print("Creating vector embeddings... (this may take a minute)")
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

# Read XML file with multiple encoding support
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

# Parse XML message to extract key information
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

# Retrieve relevant business rules using RAG
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

# Extract structured requirements from rule texts
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

# Validate message against extracted requirements
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

# Generate explanation using LLM
def generate_explanation(generation_pipeline, validation_result, message_data):
    if generation_pipeline is None:
        # Fallback if no model is available
        if validation_result['is_valid']:
            return "The message meets all requirements according to the business rules."
        else:
            missing = []
            if validation_result['missing_fields']:
                missing.append(f"Required fields: {', '.join(validation_result['missing_fields'])}")
            if validation_result['missing_attachments']:
                missing.append(f"Required attachments: {', '.join(validation_result['missing_attachments'])}")
            return f"The message was rejected because it's missing required information: {'; '.join(missing)}"
    
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
        missing_fields = ", ".join(validation_result['missing_fields']) if validation_result['missing_fields'] else "None"
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
    
    # Generate response using LLM
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

# Main validation function
def validate_xml_against_rules(xml_content, vectordb, generation_pipeline, rules_db_path):
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
    
    # Save extracted requirements to JSON file
    save_extracted_rules(
        message_data['activity_name'],
        message_data['message_type'],
        requirements,
        rules_db_path
    )
    
    print(f"\nExtracted requirements summary:")
    print(f"  Required fields: {len(requirements['required_fields'])}")
    print(f"  Required attachments: {len(requirements['required_attachments'])}")
    
    # 4. Validate the message
    print("\nStep 4: Validating message")
    validation_result = validate_message(message_data, requirements)
    
    # 5. Generate explanation with LLM
    print("\nStep 5: Generating explanation with LLM")
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
    
    print("\n========== VALIDATION COMPLETE ==========")
    return result

# Save validation result to file
def save_validation_result(result, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Validation result saved to {output_path}")

# Main function
def main():
    print("=== XML Message Validation System ===")
    print("This system validates XML messages against business rules using AI.")
    
    # Parse command line arguments
    args = setup_argparse()
    
    # Resolve file paths dynamically
    rules_path, xml_path = resolve_file_paths(args)
    
    if not rules_path or not xml_path:
        print("\nError: Could not resolve file paths. Please check your files and try again.")
        print("\nUsage examples:")
        print("  python validate.py --rules business_rules.pdf --xml message.xml")
        print("  python validate.py --interactive")
        print("  python validate.py  # Will look for files in current directory")
        return
    
    print(f"\nUsing files:")
    print(f"  Rules: {rules_path}")
    print(f"  XML: {xml_path}")
    
    # Setup language model for explanations
    print("\nInitializing AI models...")
    generation_pipeline = setup_gemma_model()
    
    # Load business rules
    print(f"\nLoading business rules from {rules_path}...")
    rules_vectordb = load_business_rules(rules_path)
    
    # Read XML message
    print(f"\nReading XML message from {xml_path}...")
    xml_content = read_xml_file(xml_path)
    
    # Validate message
    print("\nValidating message against business rules...")
    result = validate_xml_against_rules(xml_content, rules_vectordb, generation_pipeline, args.rules_db)
    
    # Save and display result
    save_validation_result(result, args.output)
    
    print("\n" + "="*50)
    print("VALIDATION RESULT")
    print("="*50)
    print(f"Decision: {result['decision']}")
    print(f"Technical Reasons: {result['technical_reasons']}")
    print(f"Explanation: {result['explanation']}")
    print("="*50)

# Entry point for script
if __name__ == "__main__":
    main()