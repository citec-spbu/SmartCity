
import argparse
import importlib

parser = argparse.ArgumentParser(description="Choose mode: RAM or Chunked")
parser.add_argument("--ram", action="store_true", help="Use train_polygon_to_buildings_RAM")
parser.add_argument("--chunked", action="store_true", help="Use train_polygon_to_buildings_CHUNKED")
args = parser.parse_args()

if args.ram and args.chunked:
    raise ValueError("Please specify only one mode: --ram or --chunked")
elif args.ram:
    module_name = "train_polygon_to_buildings_RAM"
elif args.chunked:
    module_name = "train_polygon_to_buildings_CHUNKED"
else:
    raise ValueError("Please specify a mode: --ram or --chunked")

module = importlib.import_module(module_name)

import os
import random
import json
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

from train_polygon_to_buildings_RAM import (
    DATA_FOLDER,
    MODEL_SAVE_PATH,
    DeeperMLP,
    INPUT_SIZE,
    HIDDEN_SIZE,
    OUTPUT_SIZE,
    DROPOUT_PROB,
    MAX_BUILDINGS,
    MAX_POINTS_PER_BUILDING,
    pad_or_trim_polygon_coords
)

#############################
# 1) Random Polygon
#############################

def pick_random_polygon_from_folder(folder_path):
    """
    Picks one random .json file, then one random polygon from it.
    Returns polygon_coords (list of [x,y]) in original scale, plus building data if needed.
    """
    all_json = glob.glob(os.path.join(folder_path, "*.json"))
    if not all_json:
        raise FileNotFoundError(f"No JSON files in {folder_path}")
    chosen_file = random.choice(all_json)
    print("Chosen file:", chosen_file)
    with open(chosen_file, 'r') as f:
        data = json.load(f)
    polygons = data.get("polygons_with_buildings", [])
    if not polygons:
        raise ValueError(f"No polygons found in {chosen_file}")
    poly_data = random.choice(polygons)
    polygon_coords = poly_data.get('polygon_descr', [])
    return polygon_coords

#############################
# 2) Plot Function
#############################

def plot_polygon_and_buildings(original_poly, predicted_bldgs, title="Prediction"):
    fig, ax = plt.subplots(figsize=(6,6))
    
    # Plot city polygon boundary
    if original_poly:
        x_poly = [p[0] for p in original_poly]
        y_poly = [p[1] for p in original_poly]
        # close
        if (x_poly[0], y_poly[0]) != (x_poly[-1], y_poly[-1]):
            x_poly.append(x_poly[0])
            y_poly.append(y_poly[0])
        ax.plot(x_poly, y_poly, 'b-', label="Polygon boundary")
    
    # Plot predicted building polygons
    for b_idx, bldg_pts in enumerate(predicted_bldgs):
        arr = np.array(bldg_pts)  # shape (max_points, 2)
        if (arr==0).all():
            continue
        x_b = arr[:,0]
        y_b = arr[:,1]
        # close
        if (x_b[0], y_b[0]) != (x_b[-1], y_b[-1]):
            x_b = list(x_b) + [x_b[0]]
            y_b = list(y_b) + [y_b[0]]
        ax.fill(x_b, y_b, alpha=0.3, facecolor='orange', edgecolor='red')
    
    ax.set_aspect('equal', 'datalim')
    ax.set_title(title)
    ax.legend()
    plt.show()

#############################
# 3) Main
#############################

def main():
    # Load model
    model = DeeperMLP(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        dropout_prob=DROPOUT_PROB
    )
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    print(f"Loaded model from {MODEL_SAVE_PATH} (MAX_BUILDINGS={MAX_BUILDINGS}).")
    
    # Pick random polygon
    polygon_coords_orig = pick_random_polygon_from_folder(DATA_FOLDER)
    print(f"Random polygon has {len(polygon_coords_orig)} boundary points.")
    
    # Pad/trim polygon to 5 points => flatten => [10]
    poly_in_flat = pad_or_trim_polygon_coords(polygon_coords_orig, desired_length=5)
    
    # Predict
    poly_tensor = torch.tensor(poly_in_flat, dtype=torch.float32).unsqueeze(0)  # shape [1,10]
    with torch.no_grad():
        pred = model(poly_tensor)  # shape [1, OUTPUT_SIZE], => [1, 50*10*2] = [1, 1000]
    pred = pred.squeeze(0).numpy()  # shape [1000]
    
    # Reshape => (MAX_BUILDINGS, MAX_POINTS_PER_BUILDING, 2)
    predicted_bldgs = pred.reshape(MAX_BUILDINGS, MAX_POINTS_PER_BUILDING, 2)
    
    # Plot
    plot_polygon_and_buildings(polygon_coords_orig, predicted_bldgs, 
                               title="Predicted Buildings (Up to 50)")

if __name__ == "__main__":
    main()

