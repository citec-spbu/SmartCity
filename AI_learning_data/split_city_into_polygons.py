import geopandas as gpd
from matplotlib import pyplot as plt
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point, MultiPoint
from shapely.ops import voronoi_diagram
from sklearn.cluster import KMeans
from pyproj import Transformer
import json
import os

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
def split_city_into_polygons(city_boundary, original_area, target_area, min_vertices=4, max_vertices=10):
    # handle both Polygon and MultiPolygon inputs
    if isinstance(city_boundary, MultiPolygon):
        # get the largest polygon from the MultiPolygon
        largest_polygon = max(city_boundary.geoms, key=lambda p: p.area)
        city_polygon = largest_polygon
    else:
        city_polygon = city_boundary
    
    city_area = city_polygon.area
    num_polygons = max(1, int(original_area / target_area))

    # num_polygons /= 9 # TODO: fix this, resulting polygons ~10x num_polygons

    print(f"city Area (m^2): {city_area}")
    print(f"expected area: {target_area}")
    print("expected number of polygons:", num_polygons)
    
    points = generate_random_points(city_polygon, num_points=num_polygons * 10)
    kmeans = KMeans(n_clusters=num_polygons)
    kmeans.fit(points)
    
    voronoi = voronoi_diagram(MultiPoint(points))
    result_polygons = []
    for poly in voronoi.geoms:
        clipped = poly.intersection(city_polygon)
        if isinstance(clipped, Polygon):
            simplified = simplify_polygon(clipped, min_vertices, max_vertices)
            if simplified:
                result_polygons.append(simplified)
        elif isinstance(clipped, MultiPolygon):
            for part in clipped.geoms:
                simplified = simplify_polygon(part, min_vertices, max_vertices)
                if simplified:
                    result_polygons.append(simplified)
    print("resulting number of polygons:", len(result_polygons))
    return result_polygons
def generate_random_points(polygon, num_points):
    min_x, min_y, max_x, max_y = polygon.bounds
    points = []
    while len(points) < num_points:
        point = Point(np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y))
        if polygon.contains(point):
            points.append((point.x, point.y))
    return points
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
def is_building_in_polygon(building, polygon, polygon_crs="EPSG:4979", building_crs="EPSG:4979"):
    lat = building['attributes']['latitude']
    lon = building['attributes']['longitude']
    if polygon_crs != building_crs:
        transformer = Transformer.from_crs(building_crs, polygon_crs, always_xy=True)
        lon, lat = transformer.transform(lon, lat)
    building_point = Point(lon, lat)
    return polygon.contains(building_point)
def generate_city_json(city_name, polygons, buildings_data, vertices_data, output_folder, polygon_crs, target_area):
    print("collecting buildings data for polygons...")
    result = {"polygons_with_buildings": []}
    for i, polygon in enumerate(polygons):
        print("curr pol num:", i+1)
        polygon_descr = list(polygon.exterior.coords)
        buildings_in_polygon = []
        for building_id, building in buildings_data.items():
            if is_building_in_polygon(building, polygon):
                buildings_in_polygon.append({
                    "building_id": building_id,
                    "measuredHeight": building['attributes']['measuredHeight'],
                    "area": building['attributes']['area'],
                    "latitude": building['attributes']['latitude'],
                    "longitude": building['attributes']['longitude'],
                    "boundaries": building['geometry'][0]['boundaries'][0]
                })
        result["polygons_with_buildings"].append({
            "polygon_id": i + 1,
            "polygon_descr": polygon_descr,
            "buildings": buildings_in_polygon
        })
    result["vertices"] = vertices_data
    os.makedirs(output_folder, exist_ok=True)
    size_tag = ""
    if target_area == 1000000:
        size_tag = "AVG"
    elif target_area == 100000:
        size_tag = "SMALL"
    elif target_area == 10000000:
        size_tag = "BIG"
    output_path = os.path.join(output_folder, f"{city_name}_polygons_{size_tag}.json")
    with open(output_path, 'w') as f:
        json.dump(result, f)
    print(f"JSON file saved to {output_path}")

input_dir = "AI_learning_data/downloaded_JSONs/filtered/combined"
gdf = gpd.read_file("AI_learning_data/500_Cities/CityBoundaries.shp")

for file_name in os.listdir(input_dir):
    if file_name.endswith("_COMBINED.json"):
        file_path = os.path.join(input_dir, file_name)

        city_name = file_name.split('_')[1].lower()
        filtered_gdf = gdf[gdf['NAME'].str.lower() == city_name.lower()]

        if not filtered_gdf.empty:
            city_geometry = filtered_gdf.geometry.values[0]
            original_area = city_geometry.area
            print(f"city Area (original CRS): {original_area} square meters")
            
            filtered_gdf = filtered_gdf.to_crs("EPSG:4979")
            city_geometry_epsg4979 = filtered_gdf.geometry.values[0]
            
            target_areas = [
                1_000_000,
                10_000_000,
                100_000
            ]

            print(len(target_areas))

            for target_area in target_areas:
                print("cur target_area:", target_area)
                print("splitting city into polygons...")
                polygons = split_city_into_polygons(city_geometry_epsg4979, original_area, target_area=target_area)
                print("...finished splitting city into polygons")

                # plot city split into polygons
                ax = filtered_gdf.plot(edgecolor='black', facecolor='none', figsize=(10, 8), alpha=0.5)
                for polygon in polygons:
                    gpd.GeoSeries([polygon]).plot(ax=ax, facecolor='blue', alpha=0.3, edgecolor='darkblue')

                # with open("AI_learning_data/downloaded_JSONs/filtered/combined/Alabama_Mobile_COMBINED.json", "r") as file:
                with open(file_path, "r") as file:
                    city_json_data = json.load(file)
                
                buildings_data = city_json_data['CityObjects']
                vertices_data = city_json_data['vertices']

                # # Plotting a subset of buildings
                # print("Plotting a subset of buildings...")
                # building_points = []
                # for building_id, building in buildings_data.items():
                #     # Extract building latitude and longitude
                #     lat = building['attributes']['latitude']
                #     lon = building['attributes']['longitude']
                #     building_points.append(Point(lon, lat))

                # # Plot all buildings
                # gpd.GeoSeries(building_points).plot(ax=ax, color='red', alpha=0.7, markersize=10)

                plt.title(f"random Subdivision of {city_name}")
                plt.show()

                output_folder = "AI_learning_data/learning_data"
                generate_city_json(city_name, polygons, buildings_data, vertices_data, output_folder, "EPSG:4979", target_area=target_area)
                print("...finished generate_city_json()")
        else:
            print(f"no city found with the name '{city_name}'.")
