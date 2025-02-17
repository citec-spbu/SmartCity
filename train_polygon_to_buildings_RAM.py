import os
import json
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# For geometry checks:
from shapely.geometry import Polygon
from shapely.ops import unary_union  # optional if needed

############################################
# Configuration
############################################

DATA_FOLDER = "AI_learning_data/idk_short"
MODEL_SAVE_PATH = "building_model_RAM.pth"

# We do NOT normalize coords; using them as is.

# MLP architecture
INPUT_SIZE = 10   # 5 polygon points => 10 floats for the city polygon
HIDDEN_SIZE = 512
NUM_HIDDEN_LAYERS = 4
DROPOUT_PROB = 0.2

# MAX_BUILDINGS increased to 50
MAX_BUILDINGS = 50
MAX_POINTS_PER_BUILDING = 10
OUTPUT_SIZE = MAX_BUILDINGS * MAX_POINTS_PER_BUILDING * 2  # x,y for each point

# Training
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
TEST_SPLIT_RATIO = 0.2

# Geometry penalty settings
GEOMETRY_PENALTY_WEIGHT = 1.0  # how heavily we weigh geometry violations
# You can tune that. If 0.0, geometry checks have no effect; if 10.0, geometry checks overshadow MSE.

############################################
# Utility Functions (No Normalization)
############################################

def pad_or_trim_polygon_coords(polygon_coords, desired_length=5):
    """
    Trim/pad polygon boundary to 'desired_length' points -> flatten.
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
    Return flattened (max_buildings * max_points * 2) for building boundaries.
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
            if (boundary and 
                isinstance(boundary[0], list) and 
                isinstance(boundary[0][0], (int, float))):
                for (bx, by) in boundary:
                    boundary_points.append((bx, by))
            elif (boundary and
                  isinstance(boundary[0], list) and
                  isinstance(boundary[0][0], list)):
                ring_coords = boundary[0]
                for (bx, by) in ring_coords:
                    boundary_points.append((bx, by))
            # else skip
        # pad/trim boundary points
        if len(boundary_points) > max_points:
            boundary_points = boundary_points[:max_points]
        else:
            while len(boundary_points) < max_points:
                boundary_points.append((0.0, 0.0))
        
        flattened = []
        for (bx, by) in boundary_points:
            flattened.extend([bx, by])
        
        start_idx = b_idx * max_points * 2
        for i, val in enumerate(flattened):
            output[start_idx + i] = val
    
    return output


def load_all_data_from_folder(folder_path):
    """
    Demonstrates two progress bars:
      1) For reading JSON files
      2) For processing polygons to build data_pairs
    """
    all_files = glob.glob(os.path.join(folder_path, "*.json"))
    num_files = len(all_files)
    if num_files == 0:
        print(f"No JSON files found in {folder_path}")
        return []
    
    # 1) Read each file (file-level progress bar)
    all_polygons = []
    with tqdm(total=num_files, desc="Reading JSON files") as file_bar:
        for fpath in all_files:
            with open(fpath, 'r') as f:
                jdata = json.load(f)
            polygons = jdata.get("polygons_with_buildings", [])
            all_polygons.extend(polygons)
            
            # You can print details if you like:
            print(f"\tloaded {fpath} with {len(polygons)} polygons")
            
            file_bar.update(1)
    
    # Now we have all polygons from all files in memory
    total_polygons = len(all_polygons)
    print(f"\nLoaded {total_polygons} total polygons from {num_files} files in '{folder_path}'.")

    # 2) Create data_pairs (polygon-level progress bar)
    data_pairs = []
    with tqdm(total=total_polygons, desc="Processing polygons") as poly_bar:
        for poly_data in all_polygons:
            polygon_coords = poly_data.get('polygon_descr', [])
            poly_input = pad_or_trim_polygon_coords(polygon_coords, 5)
            
            buildings = poly_data.get('buildings', [])
            building_output = pad_or_trim_building_boundaries(
                buildings,
                max_buildings=MAX_BUILDINGS,
                max_points=MAX_POINTS_PER_BUILDING
            )
            
            data_pairs.append((poly_input, building_output))
            poly_bar.update(1)
    
    return data_pairs


############################################
# Dataset & DataLoader
############################################

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


############################################
# Model Definition (Deeper + Dropout)
############################################

class DeeperMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.2):
        super(DeeperMLP, self).__init__()
        
        layers = []
        # 4 hidden layers of 512 w/ dropout
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))
        
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))
        
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))
        
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))
        
        # final
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


############################################
# Geometry-based Penalty
############################################

def geometry_penalty_loss(
    preds,           # [batch_size, OUTPUT_SIZE]
    targets,         # [batch_size, OUTPUT_SIZE]
    polygon_inputs,  # [batch_size, 10] => 5 city boundary points
    penalty_weight=1.0
):
    import numpy as np
    from shapely.geometry import Polygon
    from shapely.errors import GEOSException
    
    batch_size = preds.shape[0]
    penalty_total = 0.0
    
    for i in range(batch_size):
        # Build city polygon
        city_flat = polygon_inputs[i].tolist()  # length=10 => 5 points
        city_pts = []
        for idx in range(0, 10, 2):
            x_c = city_flat[idx]
            y_c = city_flat[idx+1]
            city_pts.append((x_c, y_c))
        city_poly = Polygon(city_pts)
        
        # Attempt to fix city polygon if invalid
        if not city_poly.is_valid:
            city_poly = city_poly.buffer(0)
        
        building_flat = preds[i].detach().cpu().numpy()
        building_reshaped = building_flat.reshape(MAX_BUILDINGS, MAX_POINTS_PER_BUILDING, 2)
        
        for b_idx in range(MAX_BUILDINGS):
            ring_coords = building_reshaped[b_idx]
            if np.all(ring_coords == 0.0):
                continue
            bldg_poly = Polygon(ring_coords)
            
            # Attempt to fix building polygon
            if not bldg_poly.is_valid:
                bldg_poly = bldg_poly.buffer(0)
            
            # If still invalid, penalize
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
                    # If intersection fails entirely, penalize
                    penalty_total += 1.0
    
    penalty_total = penalty_weight * penalty_total
    return penalty_total

############################################
# Training
############################################

def train_model():
    # 1) load data
    all_data = load_all_data_from_folder(DATA_FOLDER)
    print(f"Loaded {len(all_data)} total samples.")
    
    # 2) dataset
    dataset = BuildingsDataset(all_data)
    
    # 3) split
    test_size = int(len(dataset)*TEST_SPLIT_RATIO)
    train_size = len(dataset)-test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    print(f"Train samples: {train_size}, Test samples: {test_size}")
    
    # 4) dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)
    
    # 5) model
    model = DeeperMLP(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        dropout_prob=DROPOUT_PROB
    )
    
    # MSE + geometry penalty
    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 6) training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for poly_in, b_out in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]"):
            optimizer.zero_grad()
            preds = model(poly_in)
            # normal MSE
            loss_mse = mse_criterion(preds, b_out)
            # geometry penalty
            geom_penalty = geometry_penalty_loss(
                preds, b_out, poly_in, penalty_weight=GEOMETRY_PENALTY_WEIGHT
            )
            # total
            loss = loss_mse + geom_penalty
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * poly_in.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        
        # Evaluate
        model.eval()
        test_running_loss = 0.0
        with torch.no_grad():
            for poly_in, b_out in test_loader:
                preds = model(poly_in)
                # MSE
                loss_mse = mse_criterion(preds, b_out)
                # geometry penalty
                geom_penalty = geometry_penalty_loss(
                    preds, b_out, poly_in, penalty_weight=GEOMETRY_PENALTY_WEIGHT
                )
                # total
                loss = loss_mse + geom_penalty
                test_running_loss += loss.item() * poly_in.size(0)
        test_loss = test_running_loss / len(test_loader.dataset)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]  Train Loss: {train_loss:.6f}  |  Test Loss: {test_loss:.6f}")
    
    # 7) save
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_model()
