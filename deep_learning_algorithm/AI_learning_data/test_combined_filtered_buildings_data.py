import json
import sys
import os
import random

def construct_geometry(city_object, vertices):
    """Construct the full geometry of a building based on its vertex indices."""
    geometries = []
    for geom in city_object.get("geometry", []):
        if geom["type"] == "Solid" and "boundaries" in geom:
            solid_geometry = []
            for boundary_group in geom["boundaries"]:
                boundary_faces = []
                for face in boundary_group:
                    # Ensure `face` is a flat list of indices
                    if isinstance(face[0], list):  # Handle nested lists
                        face = [idx for sublist in face for idx in sublist]
                    boundary_faces.append([vertices[idx] for idx in face])
                solid_geometry.append(boundary_faces)
            geometries.append(solid_geometry)
    return geometries


def load_json(file_path):
    """Load a JSON file and return its content."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist.")
    with open(file_path, 'r') as file:
        return json.load(file)

def verify_building_geometries(original_file, combined_file):
    try:
        num_of_buildings = 5000

        original_data = load_json(original_file)
        combined_data = load_json(combined_file)

        original_buildings = list(original_data["CityObjects"].keys())
        original_vertices = original_data["vertices"]

        combined_buildings = combined_data["CityObjects"]
        combined_vertices = combined_data["vertices"]

        # select random buildings from the original file
        sampled_buildings = random.sample(original_buildings, min(num_of_buildings, len(original_buildings)))

        mismatches = []

        for building_id in sampled_buildings:
            if building_id not in combined_buildings:
                print(f"building ID {building_id} not found in combined file.")
                mismatches.append(building_id)
                continue

            original_geometry = construct_geometry(original_data["CityObjects"][building_id], original_vertices)
            combined_geometry = construct_geometry(combined_buildings[building_id], combined_vertices)

            # check if geometries are equal
            if original_geometry != combined_geometry:
                print(f"geometry mismatch for building ID {building_id}.")
                mismatches.append(building_id)

        if not mismatches:
            print(f"all {num_of_buildings} sampled buildings match in geometry!")
        else:
            print(f"{len(mismatches)} mismatches found. check building IDs: {mismatches}")

    except Exception as e:
        print(f"error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python3 test_combined_filtered_buildings_data.py path_to_original_file path_to_combined_file")
    else:
        verify_building_geometries(sys.argv[1], sys.argv[2])
