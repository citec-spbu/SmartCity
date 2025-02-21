import geopandas as gpd

def inspect_shapefile(shapefile_path):
    # Load the shapefile
    city_boundaries = gpd.read_file(shapefile_path)
    
    # Print column names
    print("Columns in the shapefile:", city_boundaries.columns)
    
    # Print first few rows
    print("\nFirst few rows of the dataset:")
    print(city_boundaries.head())

# Example usage
shapefile_path = "deep_learning_algorithm/AI_learning_data/500_Cities/CityBoundaries.shp"  # Update with the correct path
inspect_shapefile(shapefile_path)

