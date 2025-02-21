import geopandas as gpd
import matplotlib.pyplot as plt

def plot_mobile_city_border(shapefile_path):
    # Load the shapefile
    city_boundaries = gpd.read_file(shapefile_path)
    
    # Filter for Mobile city using 'NAME' column
    mobile_city = city_boundaries[city_boundaries['NAME'] == 'Mobile']
    
    if mobile_city.empty:
        print("Mobile city not found in the dataset.")
        return
    
    # Plot the city's border
    fig, ax = plt.subplots(figsize=(8, 8))
    mobile_city.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=2)
    ax.set_title("Borders of Mobile City")
    plt.show()

# Example usage
shapefile_path = "deep_learning_algorithm/AI_learning_data/500_Cities/CityBoundaries.shp"  # Update with the correct path
plot_mobile_city_border(shapefile_path)
