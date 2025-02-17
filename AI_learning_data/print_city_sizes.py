import geopandas as gpd
import os

gdf = gpd.read_file("AI_learning_data/500_Cities/CityBoundaries.shp")

input_dir = "AI_learning_data/city_combined_data"

for file_name in os.listdir(input_dir):
    if file_name.endswith("_COMBINED.json"):
        file_path = os.path.join(input_dir, file_name)
        city_name = file_name.split('_')[1].lower()
        print(f"working with city {city_name}")
        filtered_gdf = gdf[gdf['NAME'].str.lower() == city_name.lower()]

    if not filtered_gdf.empty:
        city_geometry = filtered_gdf.geometry.values[0]
        # original_area = city_geometry.area
        
        filtered_gdf = filtered_gdf.to_crs("EPSG:4979")
        city_geometry_epsg4979 = filtered_gdf.geometry.values[0]

        filtered_gdf = filtered_gdf.to_crs(filtered_gdf.estimate_utm_crs())  # convert to UTM
        city_geometry_projected = filtered_gdf.geometry.values[0]
        original_area = city_geometry_projected.area    
        print(f"\toriginal area: {original_area:.2f} m^2 ~ {original_area / 1e6:.2f} km^2") 