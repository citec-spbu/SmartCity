import json
import matplotlib.pyplot as plt

# A small helper function to plot a single polygon (ring).
def plot_polygon(ax, polygon_indices, vertices, color='blue'):
    """
    polygon_indices: list of vertex indices (e.g. [1, 5, 6, 2])
    vertices: global list of [lon, lat, z]
    """
    x_coords = []
    y_coords = []
    for idx in polygon_indices:
        vx, vy, vz = vertices[idx]  # (longitude, latitude, elevation)
        x_coords.append(vx)
        y_coords.append(vy)
    # Close the polygon by repeating the first vertex at the end
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])
    
    ax.plot(x_coords, y_coords, color=color)

def main():
    # Path to your JSON file
    json_path = "deep_learning_algorithm/AI_learning_data/city_combined_data/Alabama_Mobile_COMBINED.json"
    
    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    print("loaded data")
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Access the vertices array (list of [lon, lat, z])
    vertices = data["vertices"]
    
    counter = 0

    # Iterate over all buildings (CityObjects)
    for building_id, building_info in data["CityObjects"].items():
        counter += 1
        if (counter > 4_000):
            break

        # Plot the building center (longitude, latitude)
        lat = building_info["attributes"]["latitude"]
        lon = building_info["attributes"]["longitude"]
        # ax.scatter(lon, lat, color='red', s=10, zorder=2)  # Red dot for center
        
        # Each building may have one or more geometry entries
        for geom in building_info["geometry"]:
            # "boundaries" define the faces of the solid
            boundaries = geom["boundaries"]
            # boundaries is typically a list for a Solid geometry
            # For each 'shell' in boundaries
            for shell in boundaries:
                # shell is often a list of 'faces'
                # Each face can be a single ring or multiple rings
                for face in shell:
                    # face could be [[...], [...]] if there are inner rings
                    # or just [...] if there is one ring
                    if isinstance(face[0], list):
                        # multiple rings in this face
                        for ring in face:
                            plot_polygon(ax, ring, vertices, color='blue')
                    else:
                        # single ring
                        plot_polygon(ax, face, vertices, color='blue')
    
    # Labeling and aspect ratio
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Plot of Buildings and Their Centers")
    ax.set_aspect('equal', 'box')
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()

