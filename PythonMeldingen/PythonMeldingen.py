import json

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
    
    # Return the category based on the activity name, or "Onbekende categorie" if not found
    return categories.get(activity_name, "Onbekende categorie")

# Main function to process meldingen
def main():
    file_path = r"C:\Users\yassi\source\repos\PythonMeldingen\meldingen.json"
    print(f"Loading JSON file from: {file_path}")
    data = load_json(file_path)
    
    if data:
        # Navigate through the proper structure
        verzoeK_XML = data.get("verzoekXML", {})
        
        # First, check if this is a "Melding" type
        request_type = None
        if "type" in verzoeK_XML:
            request_type = verzoeK_XML["type"].get("__text")
        else:
            # Look for type in a nested structure
            for key in verzoeK_XML.keys():
                if isinstance(verzoeK_XML[key], dict) and "type" in verzoeK_XML[key]:
                    request_type = verzoeK_XML[key]["type"].get("__text")
                    break
        
        # Print the request type
        if request_type:
            print(f"Request type: {request_type}")
            
            # Only process further if it's a "Melding"
            if request_type == "Melding":
                print("This is a Melding request, processing activities...")
            elif "Aanvraag" in request_type :
                print(f"This is an Aanvraag  (type: {request_type}), skipping activity processing")
            else :
                print("this is not an melding or aanvraag. This is an rest. ")
                return
        else:
            print("Could not determine request type")
        
        # Look for container that has projectactiviteiten
        project_data = verzoeK_XML
        for key in verzoeK_XML.keys():
            if isinstance(verzoeK_XML[key], dict) and "projectactiviteiten" in verzoeK_XML[key]:
                project_data = verzoeK_XML[key]
                break
        
        # Process projectactiviteiten if found
        if "projectactiviteiten" in project_data:
            project_activiteiten = project_data["projectactiviteiten"].get("projectactiviteit", {})
            
            # Convert to list if it's a single object
            if isinstance(project_activiteiten, dict):
                project_activiteiten = [project_activiteiten]
            
            print(f"Found {len(project_activiteiten)} project activities")
            
            for activiteit in project_activiteiten:
                activiteitnaam = activiteit.get("activiteitnaam", {}).get("__text", "Unknown activity")
                categorie = categorize_activity(activiteitnaam)
                print(f"Activiteit: {activiteitnaam} -> Categorie: {categorie}")
        else:
            print("Could not locate projectactiviteiten in the JSON structure")
    else:
        print("Failed to load data from JSON.")

if __name__ == "__main__":
    main()