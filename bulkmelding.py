import os
import re
from supabase import create_client, Client
from datetime import datetime
import xml.etree.ElementTree as ET
from dotenv import load_dotenv  # ADD THIS

load_dotenv()  # ADD THIS

# Initialize Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("SUPABASE_URL and SUPABASE_KEY must be set")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def parse_xml_message(xml_content):
    """Parse XML message and extract metadata"""
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
            'activity_name': act_name,
            'message_type': msg_type,
            'specifications': specs,
            'attachments': attachments,
            'has_coordinates': has_coords,
            'verzoeknummer': verzoeknr
        }
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return None

def bulk_import_xml_messages(file_path, batch_size=100):
    """
    Import XML messages from a file where messages are separated by $
    Uses batch inserts for maximum speed
    """
    print(f"Reading file: {file_path}")
    
    # Read the entire file
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    print(f"File size: {len(content)} characters")
    
    # Split by $ delimiter
    xml_messages = content.split('$')
    xml_messages = [msg.strip() for msg in xml_messages if msg.strip()]
    
    print(f"Found {len(xml_messages)} XML messages")
    
    # Process in batches
    total_processed = 0
    total_success = 0
    total_failed = 0
    
    for i in range(0, len(xml_messages), batch_size):
        batch = xml_messages[i:i + batch_size]
        batch_records = []
        
        for idx, xml_content in enumerate(batch):
            # Skip if too short (likely not valid XML)
            if len(xml_content) < 100:
                print(f"Skipping message {i + idx + 1}: too short")
                total_failed += 1
                continue
            
            # Parse the XML
            parsed_data = parse_xml_message(xml_content)
            
            if not parsed_data:
                print(f"Failed to parse message {i + idx + 1}")
                total_failed += 1
                continue
            
            # Prepare database record
            record = {
                'xml_content': xml_content,
                'original_filename': f'bulk_import_message_{i + idx + 1}.xml',
                'verzoeknummer': parsed_data.get('verzoeknummer'),
                'activity_name': parsed_data.get('activity_name'),
                'message_type': parsed_data.get('message_type'),
                'message_metadata': parsed_data
            }
            
            batch_records.append(record)
        
        # Bulk insert this batch
        if batch_records:
            try:
                result = supabase.table('xml_messages').insert(batch_records).execute()
                batch_success = len(batch_records)
                total_success += batch_success
                total_processed += batch_success
                print(f"✓ Batch {i//batch_size + 1}: Inserted {batch_success} messages (Total: {total_success}/{len(xml_messages)})")
            except Exception as e:
                print(f"✗ Batch {i//batch_size + 1} failed: {e}")
                total_failed += len(batch_records)
                
                # Try inserting one by one as fallback
                print(f"  Retrying batch records individually...")
                for record in batch_records:
                    try:
                        supabase.table('xml_messages').insert(record).execute()
                        total_success += 1
                        total_processed += 1
                    except Exception as e2:
                        print(f"    Failed individual insert: {e2}")
                        total_failed += 1
        
        # Progress update
        if (i + batch_size) % 500 == 0:
            print(f"\nProgress: {total_processed}/{len(xml_messages)} processed, {total_success} succeeded, {total_failed} failed\n")
    
    print("\n" + "="*60)
    print(f"IMPORT COMPLETE")
    print(f"Total messages in file: {len(xml_messages)}")
    print(f"Successfully imported: {total_success}")
    print(f"Failed: {total_failed}")
    print(f"Success rate: {(total_success/len(xml_messages)*100):.1f}%")
    print("="*60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python bulk_import_xml.py <path_to_xml_file>")
        print("Example: python bulk_import_xml.py messages.txt")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    
    # Optional: customize batch size (default 100)
    batch_size = 100
    if len(sys.argv) >= 3:
        batch_size = int(sys.argv[2])
    
    print(f"Starting bulk import with batch size: {batch_size}")
    bulk_import_xml_messages(file_path, batch_size=batch_size)