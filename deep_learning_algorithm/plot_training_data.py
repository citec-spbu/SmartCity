import json
import random
import os
import argparse
import matplotlib.pyplot as plt

def plot_single_polygon(ax, polygon_data, poly_index=None):
    """
    plots one polygon (with its buildings) onto the provided Axes (ax)
    and prints debugging information to the console.
    
    Debug output includes:
      - Polygon ID
      - For each building: building number, building ID, and boundaries.
    
    Assumes the new/converted data format:
      - polygon_data['polygon_descr'] is a list of [x, y] coordinates.
      - polygon_data['buildings'] is a list of building dicts.
      - Each building has a 'boundaries' field that is either a nested list:
            [[[x1,y1], [x2,y2], ...]] or a list of [x,y] pairs.
      - Optionally, building center is stored as building['longitude'] and building['latitude'].
    """
    # --- Debug: Print Polygon Info ---
    poly_id = polygon_data.get("polygon_id", "N/A")
    if poly_index is not None:
        print(f"Polygon {poly_index+1} ID: {poly_id}")
    else:
        print("Polygon ID:", poly_id)
    
    # --- Plot the polygon boundary ---
    polygon_coords = polygon_data.get('polygon_descr', [])
    if polygon_coords:
        x_poly = [pt[0] for pt in polygon_coords]
        y_poly = [pt[1] for pt in polygon_coords]
        # Close the polygon if not already closed:
        if (x_poly[0], y_poly[0]) != (x_poly[-1], y_poly[-1]):
            x_poly.append(x_poly[0])
            y_poly.append(y_poly[0])
        ax.plot(x_poly, y_poly, 'b-', linewidth=1)
    
    # --- Process each building ---
    for b_idx, building in enumerate(polygon_data.get('buildings', [])):
        print(f"\nBuilding #{b_idx+1}")
        print("ID:", building.get("building_id", "N/A"))
        print("Boundaries:", building.get("boundaries", "None"))
        
        # Plot building center if available
        b_lon = building.get('longitude')
        b_lat = building.get('latitude')
        if b_lon is not None and b_lat is not None:
            ax.plot(b_lon, b_lat, 'ko', markersize=3)  # black circle
        
        # Plot each boundary of the building
        for boundary in building.get('boundaries', []):
            # Data may be stored as [[[x,y], ...]] or [[x,y], ...].
            if boundary and isinstance(boundary[0], list) and isinstance(boundary[0][0], list):
                ring_coords = boundary[0]
            else:
                ring_coords = boundary

            if not ring_coords:
                continue

            x_ring = [pt[0] for pt in ring_coords]
            y_ring = [pt[1] for pt in ring_coords]
            # Close the ring if necessary:
            if (x_ring[0], y_ring[0]) != (x_ring[-1], y_ring[-1]):
                x_ring.append(x_ring[0])
                y_ring.append(y_ring[0])
            ax.fill(x_ring, y_ring, alpha=0.3, facecolor='orange', edgecolor='red', linewidth=0.5)
    
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(False)  # disable grid

def plot_polygons_in_subplots(json_file, k=4):
    """
    Loads the new-format JSON file, randomly selects k polygons,
    and plots each in its own subplot (one row, k columns).
    
    Debugging information is printed to the console for each polygon.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    polygons = data.get('polygons_with_buildings', [])
    if not polygons:
        print("No polygons found in this file.")
        return
    
    # Ensure we don't try to plot more than exist.
    k = min(k, len(polygons))
    # Randomly select k polygons.
    selected_polygons = random.sample(polygons, k)

    # Create subplots (1 row, k columns)
    fig, axes = plt.subplots(1, k, figsize=(6*k, 6))
    if k == 1:
        axes = [axes]
    
    # Plot each polygon on its subplot and print debug info.
    for idx, (ax, poly_data) in enumerate(zip(axes, selected_polygons)):
        print("\n" + "="*40)
        plot_single_polygon(ax, poly_data, poly_index=idx)
    
    plt.tight_layout()
    
    # Save the figure in a 'plots' folder alongside the JSON file.
    output_folder = os.path.join(os.path.dirname(json_file), "plots")
    os.makedirs(output_folder, exist_ok=True)
    image_filename = f"new_format_{k}_subplots.png"
    output_path = os.path.join(output_folder, image_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to: {output_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Plot new-format polygons with buildings in subplots (with debugging info)."
    )
    parser.add_argument("json_file", help="Path to the new-format JSON file.")
    parser.add_argument("--k", type=int, default=4, help="Number of polygons to plot (default: 4).")
    args = parser.parse_args()

    if not os.path.isfile(args.json_file):
        print(f"Error: file not found: {args.json_file}")
        return

    plot_polygons_in_subplots(args.json_file, k=args.k)

if __name__ == "__main__":
    main()
