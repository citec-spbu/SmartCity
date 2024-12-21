import geopandas as gpd
import json
import os
from shapely.geometry import Point, Polygon, MultiPolygon
from pyproj import Transformer

PROCESSED_FILES_LOG = "AI_learning_data/filtered_files_list.json" # to skip files that have already been filtered

def load_processed_files():
    if os.path.exists(PROCESSED_FILES_LOG):
        with open(PROCESSED_FILES_LOG, "r") as log_file:
            return set(json.load(log_file))
    return set()
def save_processed_file(filename):
    processed_files = load_processed_files()
    processed_files.add(filename)
    with open(PROCESSED_FILES_LOG, "w") as log_file:
        json.dump(list(processed_files), log_file)
def transform_to_epsg_4979(geometry, original_crs):
    """Transform a shapely geometry to EPSG:4979 (latitude, longitude, and height)."""
    transformer = Transformer.from_crs(original_crs, "EPSG:4979", always_xy=True)
    if isinstance(geometry, Polygon):
        new_exterior = [transformer.transform(x, y) for x, y in geometry.exterior.coords]
        new_interiors = [
            [transformer.transform(x, y) for x, y in interior.coords]
            for interior in geometry.interiors
        ]
        return Polygon(new_exterior, new_interiors)
    elif isinstance(geometry, MultiPolygon):
        return MultiPolygon([transform_to_epsg_4979(sub_geom, original_crs) for sub_geom in geometry.geoms])
    else:
        raise ValueError("Input geometry must be a Polygon or MultiPolygon.")
def is_point_in_polygon(lon, lat, polygon):
    point = Point(lon, lat)
    return polygon.contains(point)

def filter_buildings(city_name, shapefile_path, input_folder, output_folder):
    """Filter buildings to include only those inside the boundaries of a given city."""
    gdf = gpd.read_file(shapefile_path)
    
    filtered_gdf = gdf[gdf['NAME'].str.lower() == city_name.lower()]
    if filtered_gdf.empty:
        print(f"no city found with the name '{city_name}'.")
        return

    # transform city boundary to EPSG:4979
    city_geometry = filtered_gdf.geometry.values[0]
    city_geometry_epsg4979 = transform_to_epsg_4979(city_geometry, filtered_gdf.crs)

    processed_files = load_processed_files()

    # iterate over relevant JSON files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            # Extract the second word from the filename (city name)
            try:
                state, city, *_ = filename.split("_")
            except ValueError:
                print(f"skipping malformed file: {filename}")
                continue

            if city.lower() != city_name.lower():
                continue

            # skip if the file has already been processed
            if filename in processed_files:
                print(f"file {filename} already processed. skipping.")
                continue

            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename.replace(".json", "_FILTERED.json"))

            # if os.path.exists(output_file_path):
            #     print(f"filtered file already exists for {filename}. Skipping file.")
            #     save_processed_file(filename)
            #     continue

            with open(input_file_path, "r") as file:
                buildings_data = json.load(file)
            
            filtered_buildings = {
                "CityObjects": {},
                "vertices": buildings_data.get("vertices", [])
            }

            for building_id, building in buildings_data.get("CityObjects", {}).items():
                lat = building["attributes"].get("latitude")
                lon = building["attributes"].get("longitude")

                # leave only those buildings that are inside city boundary polygon
                if lat is not None and lon is not None:
                    if is_point_in_polygon(lon, lat, city_geometry_epsg4979):
                        filtered_buildings["CityObjects"][building_id] = building

            # skip if no buildings are within the boundary
            if not filtered_buildings["CityObjects"]:
                print(f"no buildings found in {filename} for the city '{city_name}'. no file was saved.")
                save_processed_file(filename)
                continue

            # save filtered JSON
            os.makedirs(output_folder, exist_ok=True)
            with open(output_file_path, "w") as output_file:
                json.dump(filtered_buildings, output_file)

            print(f"filtered JSON saved to {output_file_path}")

            # Mark the file as processed
            save_processed_file(filename)

shapefile_path = "AI_learning_data/500_Cities/CityBoundaries.shp"
input_folder = "AI_learning_data/downloaded_JSONs"
output_folder = input_folder + "/filtered"
matching_cities_path = "AI_learning_data/matching_cities_with_fips.json"

with open(matching_cities_path, "r") as file:
    matching_cities = json.load(file)
for city_name in matching_cities.keys():
    print(f"Processing city: {city_name}")
    filter_buildings(city_name, shapefile_path, input_folder, output_folder)
