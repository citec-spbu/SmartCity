#!/usr/bin/env python3

import os
import sys
import glob
import random
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


####################################################
# 1) Constants & Model Definition
####################################################

# Adjust these to match your training setup
INPUT_SIZE = 10  # 5 polygon points => 10 floats
HIDDEN_SIZE = 512
MAX_BUILDINGS = 50
MAX_POINTS_PER_BUILDING = 10
OUTPUT_SIZE = MAX_BUILDINGS * MAX_POINTS_PER_BUILDING * 2
DROPOUT_PROB = 0.2

# If mode == 'ram', we load this file; if mode == 'chunked', we load the other:
MODEL_PATH_RAM = "building_model_RAM.pth"
MODEL_PATH_CHUNKED = "building_model_CHUNKED.pth"


class DeeperMLP(nn.Module):
    """
    A deeper feed-forward MLP with 4 hidden layers of 512 units + dropout.
    Must match the architecture used in training.
    """
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.2):
        super().__init__()
        layers = []
        # hidden layer 1
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))
        # hidden layer 2
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))
        # hidden layer 3
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))
        # hidden layer 4
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))
        # final output
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

####################################################
# 2) Helper Functions
####################################################

def pad_or_trim_polygon_coords(polygon_coords, desired_length=5):
    """
    polygon_coords: list of [x,y]
    We want exactly desired_length points => 2*desired_length floats.
    If fewer, pad with zeros. If more, truncate.
    """
    if len(polygon_coords) > desired_length:
        polygon_coords = polygon_coords[:desired_length]
    else:
        while len(polygon_coords) < desired_length:
            polygon_coords.append([0.0, 0.0])
    
    flattened = []
    for (x,y) in polygon_coords:
        flattened.extend([x,y])
    return flattened

def extract_original_buildings(poly_data):
    """
    Extract the building polygons from poly_data['buildings'] as lists of points
    so we can plot them (like the original script).
    
    Returns a list of building polygons, each being a list of (x,y).
    """
    buildings = poly_data.get('buildings', [])
    all_bldg_polys = []
    
    for b in buildings:
        boundaries = b.get('boundaries', [])
        for boundary in boundaries:
            # boundary might be [[x1,y1],[x2,y2],...] or [[[x1,y1],...]]
            if (boundary and 
                isinstance(boundary[0], list) and
                isinstance(boundary[0][0], (int,float))):
                ring_coords = boundary
            elif (boundary and
                  isinstance(boundary[0], list) and
                  isinstance(boundary[0][0], list)):
                ring_coords = boundary[0]
            else:
                ring_coords = []
            
            if ring_coords:
                all_bldg_polys.append(ring_coords)
    
    return all_bldg_polys

def convert_predicted_to_buildings(pred_array):
    """
    pred_array: shape (MAX_BUILDINGS, MAX_POINTS_PER_BUILDING, 2)
    Convert to a list of polygons (list of (x,y)) while skipping zero-padded buildings.
    """
    bldg_polys = []
    for b_idx in range(MAX_BUILDINGS):
        ring_coords = pred_array[b_idx]
        # if all zeros => skip
        if np.all(ring_coords==0.0):
            continue
        # else append
        bldg_polys.append(ring_coords.tolist())
    return bldg_polys

def plot_polygon_and_buildings(ax, polygon_coords, building_polygons, title=None):
    """
    polygon_coords: list of [x,y] for the city polygon boundary
    building_polygons: list of polygons, each is list of [x,y]
    """
    if title:
        ax.set_title(title)
    
    # Plot polygon boundary in blue
    if polygon_coords:
        x_poly = [pt[0] for pt in polygon_coords]
        y_poly = [pt[1] for pt in polygon_coords]
        # Close if needed
        if (x_poly[0], y_poly[0]) != (x_poly[-1], y_poly[-1]):
            x_poly.append(x_poly[0])
            y_poly.append(y_poly[0])
        ax.plot(x_poly, y_poly, 'b-', label="Polygon boundary")
    
    # Plot building polygons in orange
    for ring_coords in building_polygons:
        x_ring = [pt[0] for pt in ring_coords]
        y_ring = [pt[1] for pt in ring_coords]
        if (x_ring[0], y_ring[0]) != (x_ring[-1], y_ring[-1]):
            x_ring.append(x_ring[0])
            y_ring.append(y_ring[0])
        ax.fill(x_ring, y_ring, alpha=0.3, facecolor='orange', edgecolor='red', linewidth=0.5)
    
    ax.set_aspect('equal', 'datalim')
    ax.legend()
    ax.grid(False)

####################################################
# 3) Main: pick random file, pick k polygons, plot
#    2 subplots side by side:
#        Left: original polygon + original buildings
#        Right: same polygon + predicted buildings
####################################################

def main():
    parser = argparse.ArgumentParser(
        description="Pick k polygons from a random file, plot original vs predicted in 2 subplots side by side."
    )
    parser.add_argument("mode", choices=["ram","chunked"],
                        help="Mode for selecting which model to load. 'ram' -> building_model_RAM.pth, 'chunked' -> building_model_CHUNKED.pth.")
    parser.add_argument("--k", type=int, default=4,
                        help="Number of polygons to pick from the random file (default=4).")
    parser.add_argument("--plots_folder", type=str, default="plots_output",
                        help="Folder to save the output PNG plots (default='plots_output').")
    parser.add_argument("--json_folder", type=str, default="AI_learning_data/idk_short",
                        help="Folder containing .json files (default='AI_learning_data/idk_short').")
    args = parser.parse_args()

    mode = args.mode
    k = args.k
    plots_folder = args.plots_folder
    json_folder = args.json_folder

    # Determine which model file to load
    if mode == "ram":
        model_path = MODEL_PATH_RAM
    else:
        model_path = MODEL_PATH_CHUNKED
    
    print(f"Using mode='{mode}' => loading model from '{model_path}'")
    print(f"Will pick {k} polygons from a random file in '{json_folder}'")
    print(f"Saving figures to '{plots_folder}' with dpi=600")

    # Create output folder
    os.makedirs(plots_folder, exist_ok=True)

    # 1) Load the model
    model = DeeperMLP(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        dropout_prob=DROPOUT_PROB
    )
    if not os.path.exists(model_path):
        print(f"ERROR: model file '{model_path}' not found!")
        sys.exit(1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Loaded model from {model_path}")

    # 2) Pick a random JSON file
    all_json_files = glob.glob(os.path.join(json_folder, "*.json"))
    if not all_json_files:
        print(f"No JSON files found in '{json_folder}'. Exiting.")
        sys.exit(0)
    chosen_file = random.choice(all_json_files)
    print(f"Chosen random file: {chosen_file}")

    # 3) Load polygons from that file
    with open(chosen_file, 'r') as f:
        data = json.load(f)
    polygons = data.get("polygons_with_buildings", [])
    if not polygons:
        print(f"No polygons in file '{chosen_file}'. Exiting.")
        sys.exit(0)

    # 4) Randomly pick k polygons
    if len(polygons) <= k:
        selected_polygons = polygons
    else:
        selected_polygons = random.sample(polygons, k)
    
    print(f"Selected {len(selected_polygons)} polygons from file.")

    # 5) For each polygon, create a figure with 2 subplots:
    #    Left => original
    #    Right => predicted
    for idx, poly_data in enumerate(selected_polygons):
        # Original polygon boundary
        polygon_coords = poly_data.get('polygon_descr', [])
        original_poly = polygon_coords[:]  # copy for plotting

        # Original building boundaries
        original_bldg_list = extract_original_buildings(poly_data)
        
        # Prepare input for model
        poly_in_flat = pad_or_trim_polygon_coords(polygon_coords, desired_length=5)
        poly_tensor = torch.tensor(poly_in_flat, dtype=torch.float32).unsqueeze(0)  # [1,10]

        # Inference
        with torch.no_grad():
            preds = model(poly_tensor)  # shape [1, OUTPUT_SIZE]
        preds = preds.squeeze(0).numpy()  # shape [OUTPUT_SIZE]
        preds_3d = preds.reshape(MAX_BUILDINGS, MAX_POINTS_PER_BUILDING, 2)
        predicted_bldg_list = convert_predicted_to_buildings(preds_3d)

        # Plot side by side
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12,6))
        
        # Left => original
        plot_polygon_and_buildings(ax_left, original_poly, original_bldg_list,
                                   title=f"Polygon #{idx+1}: Original")
        # Right => predicted
        plot_polygon_and_buildings(ax_right, original_poly, predicted_bldg_list,
                                   title=f"Polygon #{idx+1}: Predicted")

        # Save
        out_filename = f"poly_{idx+1}_of_{len(selected_polygons)}.png"
        out_path = os.path.join(plots_folder, out_filename)
        plt.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
