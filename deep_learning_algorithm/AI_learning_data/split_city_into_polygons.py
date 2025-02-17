import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point, MultiPoint
from shapely.ops import voronoi_diagram
from sklearn.cluster import MiniBatchKMeans
from pyproj import Transformer
import json
import os
import time
from rtree import index
from collections import defaultdict
from shapely.prepared import prep
from joblib import Parallel, delayed

def transform_to_epsg_4979(polygon, original_crs):
    transformer = Transformer.from_crs(original_crs, "EPSG:4979", always_xy=True)
    if isinstance(polygon, Polygon):
        new_exterior = [transformer.transform(x, y) for x, y in polygon.exterior.coords]
        new_interiors = [
            [transformer.transform(x, y) for x, y in interior.coords]
            for interior in polygon.interiors
        ]
        return Polygon(new_exterior, new_interiors)
    elif isinstance(polygon, MultiPolygon):
        transformed_polygons = [
            transform_to_epsg_4979(sub_polygon, original_crs)
            for sub_polygon in polygon.geoms
        ]
        return MultiPolygon(transformed_polygons)
    else:
        raise ValueError("Input geometry must be a Polygon or MultiPolygon.")

def process_polygon(poly, city_polygon, min_vertices, max_vertices):
    clipped = poly.intersection(city_polygon)
    if isinstance(clipped, Polygon):
        return simplify_polygon(clipped, min_vertices, max_vertices)
    elif isinstance(clipped, MultiPolygon):
        return [simplify_polygon(part, min_vertices, max_vertices) for part in clipped.geoms]
    return None
def split_city_into_polygons(city_boundary, original_area, target_area, min_vertices=4, max_vertices=10):
    # handle both Polygon and MultiPolygon inputs
    if isinstance(city_boundary, MultiPolygon):
        # get the largest polygon from the MultiPolygon
        largest_polygon = max(city_boundary.geoms, key=lambda p: p.area)
        city_polygon = largest_polygon
    else:
        city_polygon = city_boundary
    
    num_polygons = max(1, int(original_area / target_area))

    print(f"city area (m^2): {original_area}")
    print(f"expected area: {target_area}")
    print("expected number of polygons:", num_polygons)
    
    points = generate_random_points(city_polygon, num_points=num_polygons)
    kmeans = MiniBatchKMeans(n_clusters=num_polygons, batch_size=256, random_state=42, n_init='auto')
    kmeans.fit(points)
    
    centroids = kmeans.cluster_centers_
    voronoi = voronoi_diagram(MultiPoint([Point(c) for c in centroids]))

    # parallelized polygon processing
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_polygon)(poly, city_polygon, min_vertices, max_vertices) 
        for poly in voronoi.geoms
    )

    # flatten list and remove None values
    result_polygons = [
        poly for sublist in results if sublist 
        for poly in (sublist if isinstance(sublist, list) else [sublist])
    ]

    print(f"resulting number of polygons: {len(result_polygons)}")
    return result_polygons
def generate_random_points(polygon, num_points):
    min_x, min_y, max_x, max_y = polygon.bounds
    points = []
    
    while len(points) < num_points:
        # generate in batches to improve efficiency
        batch_size = (num_points - len(points)) * 2
        candidate_points = np.column_stack((
            np.random.uniform(min_x, max_x, batch_size),
            np.random.uniform(min_y, max_y, batch_size)
        ))

        # convert numpy array to shapely points and filter inside polygon
        valid_points = [Point(p) for p in candidate_points if polygon.contains(Point(p))]
        points.extend(valid_points[:num_points - len(points)])  # ensure we don't exceed num_points
    
    return np.array([[p.x, p.y] for p in points])
def simplify_polygon(polygon, min_vertices, max_vertices):
    exterior = list(polygon.exterior.coords)
    if len(exterior) < min_vertices:
        return None
    elif len(exterior) > max_vertices:
        step = len(exterior) // max_vertices
        simplified = exterior[::step]
        if len(simplified) < min_vertices:
            simplified = exterior[:min_vertices]
    else:
        simplified = exterior
    return Polygon(simplified)

def extract_pols_with_buildings_to_json(city_name, polygons, buildings_data, output_folder, target_area):
    print("preprocessing buildings and polygons...")
    
    # precompute building points
    building_points = {}
    for building_id, data in buildings_data.items():
        lat = data['attributes']['latitude']
        lon = data['attributes']['longitude']
        building_points[building_id] = Point(lon, lat)  # x=longitude, y=latitude
    
    # create R-tree index for polygons
    polygon_index = index.Index()
    prepared_polygons = []
    for idx, polygon in enumerate(polygons):
        polygon_index.insert(idx, polygon.bounds)
        prepared_polygons.append(prep(polygon))  # Prepare for faster contains checks
    
    # map buildings to polygons
    print("mapping buildings to polygons...")
    polygon_building_map = defaultdict(list)
    
    for building_id, point in building_points.items():
        # find candidate polygons using spatial index
        candidates = polygon_index.intersection((point.x, point.y, point.x, point.y))
        for candidate_idx in candidates:
            if prepared_polygons[candidate_idx].contains(point):
                polygon_building_map[candidate_idx].append(building_id)
    
    # build result structure
    print("building result structure...")
    result = {"polygons_with_buildings": []}
    for polygon_idx, building_ids in polygon_building_map.items():
        polygon = polygons[polygon_idx]
        buildings_in_polygon = []
        for bid in building_ids:
            building = buildings_data[bid]
            buildings_in_polygon.append({
                "building_id": bid,
                "measuredHeight": building['attributes']['measuredHeight'],
                "area": building['attributes']['area'],
                "latitude": building['attributes']['latitude'],
                "longitude": building['attributes']['longitude'],
                "boundaries": building['geometry'][0]['boundaries'][0]
            })
        
        result["polygons_with_buildings"].append({
            "polygon_id": polygon_idx + 1,
            "polygon_descr": list(polygon.exterior.coords),
            "buildings": buildings_in_polygon
        })
    
    os.makedirs(output_folder, exist_ok=True)
    formatted_area = f"{int(target_area)}"
    output_path = os.path.join(output_folder, f"{city_name}_polygons_{formatted_area}.json")
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=None, separators=(',', ':'))
    print(f"JSON file saved to {output_path}")

input_dir = "AI_learning_data/city_combined_data"
output_folder = "AI_learning_data/idk"
gdf = gpd.read_file("AI_learning_data/500_Cities/CityBoundaries.shp")

# all in m^2
small_areas = list(range(5_000, 25_001, 1_000))
medium_areas = list(range(25_000, 50_001, 5_000))
large_areas = list(range(50_000, 150_001, 10_000))
target_areas = small_areas + medium_areas + large_areas

small_cities = ["passaic"]                                          # may not be very informative
huge_cities = ["anchorage", "los angeles", "san diego", "honolulu"] # take several hours to get 5,000 m^2 split

for file_name in os.listdir(input_dir):
    if file_name.endswith("_COMBINED.json"):
        file_path = os.path.join(input_dir, file_name)
        city_name = file_name.split('_')[1].lower()
        print(f"working with city {city_name}")
        filtered_gdf = gdf[gdf['NAME'].str.lower() == city_name.lower()]

        if city_name in huge_cities or city_name in small_cities:
            continue

        if not filtered_gdf.empty:
            city_geometry = filtered_gdf.geometry.values[0]
            
            filtered_gdf = filtered_gdf.to_crs("EPSG:4979")
            city_geometry_epsg4979 = filtered_gdf.geometry.values[0]

            filtered_gdf = filtered_gdf.to_crs(filtered_gdf.estimate_utm_crs())  # convert to UTM
            city_geometry_projected = filtered_gdf.geometry.values[0]
            original_area = city_geometry_projected.area    
            print(f"\toriginal area: {original_area:.2f} m^2 ~ {original_area / 1e6:.2f} km^2") 

            with open(file_path, 'r') as file:
                city_json_data = json.load(file)
            buildings_data = city_json_data['CityObjects']
            vertices_filename = os.path.join(output_folder, f"{city_name}_vertices.json")
            if not os.path.exists(vertices_filename):
                vertices_data = city_json_data['vertices']
                with open(vertices_filename, 'w') as v_file:
                    json.dump(vertices_data, v_file, indent=None, separators=(',', ':'))
            
            for target_area in target_areas:
                # check if output file already exists
                output_filename = os.path.join(
                    output_folder, 
                    f"{city_name}_polygons_{target_area}.json"
                )
                if os.path.exists(output_filename):
                    print(f"\tfile {output_filename} exists. skipping.")
                    continue

                print("\tcur target_area:", f"{target_area:,}")
                print("\t\tsplitting city into polygons...")
                start_time = time.time()
                polygons = split_city_into_polygons(city_geometry_epsg4979, original_area, target_area=target_area)
                old_duration = time.time() - start_time
                print(f"\t\t...finished splitting city into polygons, it took {time.time() - start_time:.2f} seconds")
                
                print("\textracting polygons with buildings to json file...")
                extract_pols_with_buildings_to_json(city_name, polygons, buildings_data, output_folder, target_area=target_area)
                print("...finished extracting polygons with buildings to json file")
        else:
            print(f"no city found with the name '{city_name}'.")
