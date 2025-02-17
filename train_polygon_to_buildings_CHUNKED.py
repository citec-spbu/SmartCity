import os
import glob
import math
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# For geometry checks:
from shapely.geometry import Polygon
from shapely.errors import GEOSException

#########################################
# Configuration
#########################################

# Folders & paths
DATA_FOLDER = "AI_learning_data/idk_short"  # or "AI_learning_data/idk_short", etc.
MODEL_SAVE_PATH = "building_model_chunked.pth"

# Model architecture
INPUT_SIZE = 10   # e.g. 5 polygon boundary points -> 10 floats
HIDDEN_SIZE = 512
NUM_HIDDEN_LAYERS = 4
DROPOUT_PROB = 0.2

# Max buildings, etc.
MAX_BUILDINGS = 50
MAX_POINTS_PER_BUILDING = 10
OUTPUT_SIZE = MAX_BUILDINGS * MAX_POINTS_PER_BUILDING * 2  # up to 50 bldgs * 10 pts * (x,y)

# Training hyperparams
BATCH_SIZE = 4
LEARNING_RATE = 1e-3

# We'll do EPOCHS passes over the entire file list
EPOCHS = 1   # increase if you want more passes
FILES_PER_CHUNK = 10  # how many .json files to load at a time
SHUFFLE_FILES = True  # if True, shuffle the file list each epoch

# Geometry penalty
GEOMETRY_PENALTY_WEIGHT = 1.0  # penalty weight for invalid polygons
# If training is too slow with geometry checks, reduce or remove.

#########################################
# Utility: Pad/Trim Polygon + Buildings
#########################################

def pad_or_trim_polygon_coords(polygon_coords, desired_length=5):
    """
    polygon_coords: list of [x,y]
    We want exactly desired_length points -> 2*desired_length floats.
    If fewer, pad with zeros; if more, truncate.
    """
    if len(polygon_coords) > desired_length:
        polygon_coords = polygon_coords[:desired_length]
    else:
        while len(polygon_coords) < desired_length:
            polygon_coords.append([0.0, 0.0])
    
    flattened = []
    for (x, y) in polygon_coords:
        flattened.extend([x, y])
    return flattened


def pad_or_trim_building_boundaries(buildings, max_buildings=50, max_points=10):
    """
    For each building, gather boundary points, pad/trim to max_points, flatten.
    Then pad/trim buildings to max_buildings.
    Return a single list of length max_buildings*max_points*2.
    """
    if len(buildings) > max_buildings:
        buildings = buildings[:max_buildings]
    
    total_length = max_buildings * max_points * 2
    output = [0.0]*total_length
    
    for b_idx, b in enumerate(buildings):
        if b_idx >= max_buildings:
            break
        boundary_points = []
        for boundary in b.get('boundaries', []):
            # boundary might be [[x,y]...] or [[[x,y],[x,y],...]]
            if (boundary and
                isinstance(boundary[0], list) and
                isinstance(boundary[0][0], (int,float))):
                # simple
                for (bx, by) in boundary:
                    boundary_points.append((bx, by))
            elif (boundary and
                  isinstance(boundary[0], list) and
                  isinstance(boundary[0][0], list)):
                # nested
                ring_coords = boundary[0]
                for (bx, by) in ring_coords:
                    boundary_points.append((bx, by))
            # else skip if empty
        
        # pad/trim boundary_points
        if len(boundary_points) > max_points:
            boundary_points = boundary_points[:max_points]
        else:
            while len(boundary_points) < max_points:
                boundary_points.append((0.0, 0.0))
        
        # flatten
        flattened_build = []
        for (bx, by) in boundary_points:
            flattened_build.extend([bx, by])
        
        start_idx = b_idx * max_points * 2
        for i, val in enumerate(flattened_build):
            output[start_idx + i] = val
    
    return output


#########################################
# Chunked Loading
#########################################

def load_data_from_files(file_list):
    """
    Reads polygon data from the given file_list (list of .json paths).
    Returns a list of (poly_in, b_out) pairs.
    """
    data_pairs = []
    for fpath in file_list:
        with open(fpath, 'r') as f:
            jdata = json.load(f)
        polygons = jdata.get("polygons_with_buildings", [])
        for poly_data in polygons:
            polygon_coords = poly_data.get('polygon_descr', [])
            poly_input = pad_or_trim_polygon_coords(polygon_coords, desired_length=5)
            
            buildings = poly_data.get('buildings', [])
            building_out = pad_or_trim_building_boundaries(
                buildings,
                max_buildings=MAX_BUILDINGS,
                max_points=MAX_POINTS_PER_BUILDING
            )
            data_pairs.append((poly_input, building_out))
    return data_pairs


#########################################
# Dataset
#########################################

class BuildingsDataset(Dataset):
    def __init__(self, data_pairs):
        super().__init__()
        self.data_pairs = data_pairs
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        poly_in, b_out = self.data_pairs[idx]
        poly_tensor = torch.tensor(poly_in, dtype=torch.float32)
        b_tensor    = torch.tensor(b_out, dtype=torch.float32)
        return poly_tensor, b_tensor


#########################################
# Model
#########################################

class DeeperMLP(nn.Module):
    """
    4 hidden layers of 512, dropout, final layer => OUTPUT_SIZE
    """
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.2):
        super().__init__()
        
        layers = []
        # hidden1
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))
        # hidden2
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))
        # hidden3
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))
        # hidden4
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))
        # output
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


#########################################
# Geometry Penalty
#########################################

def geometry_penalty_loss(preds, polygon_inputs, penalty_weight=1.0):
    """
    Adds penalty if a predicted building polygon is:
      - self-intersecting (invalid)
      - fully outside the city polygon (intersection area=0 but building area>0)
    
    preds: shape [batch_size, OUTPUT_SIZE]
    polygon_inputs: shape [batch_size, 10] (assuming 5 city points -> 10 floats)
    penalty_weight: scaling factor
    """
    import numpy as np
    
    batch_size = preds.shape[0]
    penalty_total = 0.0
    
    for i in range(batch_size):
        # 1) Build the city polygon from polygon_inputs
        city_flat = polygon_inputs[i].tolist()  # length=10
        city_pts = []
        for idx in range(0,10,2):
            x_c = city_flat[idx]
            y_c = city_flat[idx+1]
            city_pts.append((x_c, y_c))
        city_poly = Polygon(city_pts)
        # Attempt to fix city if invalid
        if not city_poly.is_valid:
            city_poly = city_poly.buffer(0)
        
        # 2) Reshape predicted building polygons
        building_flat = preds[i].detach().cpu().numpy()  # shape [OUTPUT_SIZE]
        building_reshaped = building_flat.reshape(MAX_BUILDINGS, MAX_POINTS_PER_BUILDING, 2)
        
        for b_idx in range(MAX_BUILDINGS):
            ring_coords = building_reshaped[b_idx]
            # skip if all zero => padded building
            if np.all(ring_coords==0.0):
                continue
            
            bldg_poly = Polygon(ring_coords)
            # Attempt to fix building
            if not bldg_poly.is_valid:
                bldg_poly = bldg_poly.buffer(0)
            
            # If still invalid => penalty
            if not bldg_poly.is_valid:
                penalty_total += 1.0
            else:
                # Intersection check
                try:
                    inter_area = bldg_poly.intersection(city_poly).area
                    bldg_area  = bldg_poly.area
                    if bldg_area > 0 and inter_area == 0:
                        # fully outside
                        penalty_total += 1.0
                except GEOSException:
                    # If intersection fails, also penalize
                    penalty_total += 1.0
    
    return penalty_weight * penalty_total


#########################################
# Training with Chunking
#########################################

def train_model():
    all_files = glob.glob(os.path.join(DATA_FOLDER, "*.json"))
    print(f"found {len(all_files)} .json files in {DATA_FOLDER}.")
    
    # define model, loss, optimizer
    model = DeeperMLP(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        dropout_prob=DROPOUT_PROB
    )
    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    num_chunks = math.ceil(len(all_files)/FILES_PER_CHUNK)
    
    for epoch in range(EPOCHS):
        print(f"\n=== epoch {epoch+1}/{EPOCHS} ===")
        
        # shuffle file list each epoch, if desired
        if SHUFFLE_FILES:
            random.shuffle(all_files)
        
        for chunk_i in range(num_chunks):
            # pick subset of files for this chunk
            start_idx = chunk_i*FILES_PER_CHUNK
            end_idx   = (chunk_i+1)*FILES_PER_CHUNK
            chunk_files = all_files[start_idx:end_idx]
            
            print(f"\tloading chunk {chunk_i+1}/{num_chunks} with {len(chunk_files)} files...")
            data_pairs = load_data_from_files(chunk_files)
            if len(data_pairs) == 0:
                print("\t\tno polygons in this chunk, skipping.")
                continue
            
            dataset = BuildingsDataset(data_pairs)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            # train on this chunk (mini-epoch)
            model.train()
            running_loss = 0.0
            for poly_in, b_out in tqdm(loader, desc=f"chunk {chunk_i+1}/{num_chunks}"):
                optimizer.zero_grad()
                
                preds = model(poly_in)
                # MSE part
                loss_mse = mse_criterion(preds, b_out)
                # geometry penalty
                geom_pen = geometry_penalty_loss(
                    preds, poly_in, penalty_weight=GEOMETRY_PENALTY_WEIGHT
                )
                loss = loss_mse + geom_pen
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * poly_in.size(0)
            
            if len(loader.dataset) > 0:
                chunk_loss = running_loss / len(loader.dataset)
            else:
                chunk_loss = 0.0
            
            print(f"\t\tChunk {chunk_i+1}/{num_chunks} train_loss = {chunk_loss:.6f}")
            
            # free memory
            del data_pairs, dataset, loader
        
        # Optionally save after each epoch
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"model saved to {MODEL_SAVE_PATH}")
    
    print("training complete.")


if __name__ == "__main__":
    print("train_model()...")
    train_model()
    print("...train_model()")
