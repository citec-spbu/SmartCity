import json
import glob
import os
from collections import defaultdict
from tqdm import tqdm

def merge_city_jsons_per_city(input_folder, combined_files_list):
    combined_folder = os.path.join(input_folder, "combined")
    os.makedirs(combined_folder, exist_ok=True)
    # combined_files_list = os.path.join(input_folder, combined_files_list)

    # load list of already combined files if it exists
    if os.path.exists(combined_files_list):
        with open(combined_files_list, 'r') as f:
            combined_files = set(json.load(f))
    else:
        combined_files = set()

    print("combined_files_list:", combined_files_list)
    print("combined_files:", combined_files)

    # group files by city prefix
    city_groups = defaultdict(list)
    json_files = glob.glob(f"{input_folder}/*.json")
    for file in json_files:
        filename = os.path.basename(file)
        # extract city prefix (everything before the last dash-separated part)
        city_prefix = "-".join(filename.split("-")[:-1]).rsplit("_", 1)[0]
        city_groups[city_prefix].append(file)

    # track combined files during the process
    cities_to_combine = list(city_groups.keys())
    
    for city_prefix in tqdm(cities_to_combine, desc="Merging Cities", ncols=100):
        combined_file_name = f"{city_prefix}_COMBINED.json"
        
        if combined_file_name in combined_files:
            continue # skip if already combined

        combined_data = {
            "CityObjects": {},
            "vertices": []
        }
        vertex_offset = 0

        # if there's only one file, copy it directly to the combined data
        if len(city_groups[city_prefix]) == 1:
            single_file = city_groups[city_prefix][0]
            with open(single_file, 'r') as f:
                data = json.load(f)

            combined_data["CityObjects"] = data["CityObjects"]
            combined_data["vertices"] = data["vertices"]
        else:
            for file in city_groups[city_prefix]:
                with open(file, 'r') as f:
                    data = json.load(f)

                # merge CityObjects
                for key, value in data["CityObjects"].items():
                    # adjust geometry indices to account for offset in vertices
                    for geom in value["geometry"]:
                        if geom["type"] == "Solid" and "boundaries" in geom:
                            new_boundaries = []
                            for shell in geom["boundaries"]:  # shell level
                                new_shell = []
                                for face in shell:  # face level
                                    new_face = []
                                    for ring in face:  # ring level (exterior/interior)
                                        new_ring = []
                                        for index in ring:  # vertex indices level
                                            if not isinstance(index, int):
                                                raise TypeError(f"Face indices must be integers, got {type(index)}")
                                            new_ring.append(index + vertex_offset)
                                        new_face.append(new_ring)
                                    new_shell.append(new_face)
                                new_boundaries.append(new_shell)
                            geom["boundaries"] = new_boundaries

                    combined_data["CityObjects"][key] = value

                combined_data["vertices"].extend(data["vertices"])
                vertex_offset += len(data["vertices"])

        output_file = os.path.join(combined_folder, combined_file_name)
        with open(output_file, 'w') as f:
            json.dump(combined_data, f)

        combined_files.add(combined_file_name)
        
        with open(combined_files_list, 'w') as f:
            json.dump(list(combined_files), f, indent=4)

        print(f"Combined file created: {output_file}")

input_folder = "AI_learning_data/downloaded_JSONs/filtered"
combined_files_log = "AI_learning_data/downloaded_JSONs/filtered/combined/combined_files_log.json" 
merge_city_jsons_per_city(input_folder, combined_files_log)
