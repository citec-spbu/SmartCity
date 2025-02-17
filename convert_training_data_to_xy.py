import os
import json
import numpy as np

def transform_polygon(polygon_coords):
    """
    Given a list of [lon, lat] points, compute the centroid and return:
      - the list of points shifted so the centroid is at (0,0)
      - the centroid itself (as [lon, lat])
    """
    arr = np.array(polygon_coords)  # shape (N,2)
    centroid = arr.mean(axis=0)
    transformed = (arr - centroid).tolist()
    return transformed, centroid.tolist()

def transform_building(building, centroid, vertices):
    """
    Given a building dictionary, subtract the polygon centroid from:
      - the building's (longitude, latitude)
      - each building boundary (using indices from the vertices file)
    
    The vertices file is assumed to be a list of [lon, lat, z] triplets.
    We drop the z value.
    """
    # Transform the building’s central location.
    building['longitude'] = building['longitude'] - centroid[0]
    building['latitude']  = building['latitude'] - centroid[1]
    
    new_boundaries = []
    for boundary in building.get('boundaries', []):
        # Some boundaries are nested (e.g. [[indices]]); flatten if needed.
        if boundary and isinstance(boundary[0], list):
            indices = boundary[0]
        else:
            indices = boundary
        transformed_ring = []
        for idx in indices:
            try:
                # Get vertex coordinate; use only lon and lat (drop altitude)
                vertex = vertices[idx]
            except IndexError:
                print(f"Warning: vertex index {idx} out of range. Skipping.")
                continue
            transformed_point = [vertex[0] - centroid[0], vertex[1] - centroid[1]]
            transformed_ring.append(transformed_point)
        new_boundaries.append(transformed_ring)
    building['boundaries'] = new_boundaries
    return building

def process_polygon_file(polygon_filepath, vertices_filepath):
    # Load the vertices file (list of [lon, lat, z] points)
    with open(vertices_filepath, 'r') as f:
        vertices = json.load(f)
    
    # Load the polygon file
    with open(polygon_filepath, 'r') as f:
        data = json.load(f)
    
    # Process each polygon in the file
    for poly in data.get("polygons_with_buildings", []):
        # Transform the polygon's coordinates and compute centroid
        new_polygon, centroid = transform_polygon(poly["polygon_descr"])
        poly["polygon_descr"] = new_polygon
        
        # Process each building inside the polygon
        for building in poly.get("buildings", []):
            transform_building(building, centroid, vertices)
    
    # Write the transformed data to a new file (appending '_converted')
    new_filename = polygon_filepath.replace(".json", "_converted.json")
    with open(new_filename, 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    print(f"Processed and saved: {new_filename}")
    
    # Remove the original polygon file after conversion
    os.remove(polygon_filepath)
    print(f"Removed original file: {polygon_filepath}")

def main():
    folder = "AI_learning_data/idk"
    files = os.listdir(folder)
    
    # Process each file that is a polygon file and does not already contain 'converted'
    for filename in files:
        if "polygons" in filename and filename.endswith(".json") and "converted" not in filename:
            polygon_filepath = os.path.join(folder, filename)
            # Identify the corresponding vertices file based on filename prefix.
            # E.g., for "lynchburg_polygons_25000.json", the prefix is "lynchburg"
            prefix = filename.split("_polygons")[0]
            vertices_filename = prefix + "_vertices.json"
            vertices_filepath = os.path.join(folder, vertices_filename)
            
            if os.path.exists(vertices_filepath):
                print("Working with polygon file:", polygon_filepath)
                process_polygon_file(polygon_filepath, vertices_filepath)
            else:
                print(f"Warning: Vertices file not found for {polygon_filepath}")

if __name__ == "__main__":
    main()
