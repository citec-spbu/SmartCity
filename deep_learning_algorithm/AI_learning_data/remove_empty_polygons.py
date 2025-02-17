import os
import json

folder_path = "AI_learning_data/learning_data"  # Replace with the path to your folder

# Iterate through all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".json"):
        file_path = os.path.join(folder_path, file_name)

        # Load the JSON file
        with open(file_path, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {file_name}: {e}")
                continue

        # Remove polygons with empty "buildings"
        if "polygons_with_buildings" in data:
            data["polygons_with_buildings"] = [
                polygon for polygon in data["polygons_with_buildings"] if polygon.get("buildings")
            ]

        # Save the cleaned data back to the file
        with open(file_path, "w") as file:
            json.dump(data, file)

        print(f"processed file: {file_name}")

print("all files processed.")