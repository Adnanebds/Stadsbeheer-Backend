import json
from supabase import create_client, Client
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Initialize Supabase
url = "https://yqlpqgsoynbtvhwprjyt.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxbHBxZ3NveW5idHZod3Byanl0Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NDg5MzUzNiwiZXhwIjoyMDYwNDY5NTM2fQ.fknznZ4sWszgZCUj_RgTenxw6r9u1NZqm_lvvmXll8A"
supabase: Client = create_client(url, key)

# Function to load the JSON file
def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print("JSON file loaded successfully.")
            return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")

# Categorizing the activities
def categorize_activity(activity_name):
    categories = {
        "Toepassen van grond of baggerspecie op of in de landbodem": "Werkverdeler bbk",
        "Slopen van een bouwwerk of gedeelte daarvan of asbest verwijderen": "MO sloop/asbest",
        "Graven in bodem met een kwaliteit onder of gelijk aan de interventiewaarde bodemkwaliteit": "Werkverdeler bbk",
        "Saneren van de bodem": "Werkverdeler bodem",
        "Graven in bodem met een kwaliteit boven de interventiewaarde bodemkwaliteit": "Werkverdeler bodem",
        "Op de landbodem opslaan, zeven, mechanisch ontwateren of samenvoegen van zonder bewerking herbruikbare grond of baggerspecie": "Werkverdeler bbk",
        "Aanvraag vergunning": "Quickscan team",
        "Bunkerstation of andere tankplaats voor schepen": "Werkverdeler vergunningverlening industrie"
    }
    return categories.get(activity_name, "Onbekende categorie")

# Save to Supabase
def save_to_supabase(request_type, subtype):
    data = {"type": request_type, "subtype": subtype}
    response = supabase.table("Messages").insert(data).execute()
    print("Saved to Supabase:", response)

# Main function to process meldingen
def main():
    file_path = r"C:\Users\kenki\williamprojectplan\stadsbeheerbackend\meldingen.json"
    print(f"Loading JSON file from: {file_path}")
    data = load_json(file_path)
    
    if data:
        verzoeK_XML = data.get("verzoekXML", {})

        request_type = None
        if "type" in verzoeK_XML:
            request_type = verzoeK_XML["type"].get("__text")
        else:
            for key in verzoeK_XML.keys():
                if isinstance(verzoeK_XML[key], dict) and "type" in verzoeK_XML[key]:
                    request_type = verzoeK_XML[key]["type"].get("__text")
                    break
        
        if request_type:
            print(f"Request type: {request_type}")
            if request_type == "Melding":
                print("This is a Melding request, processing activities...")
            elif "Aanvraag" in request_type:
                print(f"This is an Aanvraag (type: {request_type}), skipping activity processing")
                return
            else:
                print("Not a Melding or Aanvraag.")
                return
        else:
            print("Could not determine request type")
            return

        project_data = verzoeK_XML
        for key in verzoeK_XML.keys():
            if isinstance(verzoeK_XML[key], dict) and "projectactiviteiten" in verzoeK_XML[key]:
                project_data = verzoeK_XML[key]
                break

        if "projectactiviteiten" in project_data:
            project_activiteiten = project_data["projectactiviteiten"].get("projectactiviteit", {})
            if isinstance(project_activiteiten, dict):
                project_activiteiten = [project_activiteiten]
            
            print(f"Found {len(project_activiteiten)} project activities")
            for activiteit in project_activiteiten:
                activiteitnaam = activiteit.get("activiteitnaam", {}).get("__text", "Unknown activity")
                categorie = categorize_activity(activiteitnaam)
                print(f"Activiteit: {activiteitnaam} -> Categorie: {categorie}")
                save_to_supabase(request_type, categorie)
        else:
            print("Could not locate projectactiviteiten in the JSON structure")
    else:
        print("Failed to load data from JSON.")

if __name__ == "__main__":
    main()
