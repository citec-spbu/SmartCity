import os
import json
import requests
from tqdm import tqdm

matching_cities_path = 'AI_learning_data/matching_cities_with_fips.json'
all_OCM_site_tree_path = 'AI_learning_data/opencitymodel/all_OCM_site_tree.json'
download_dir = 'AI_learning_data/downloaded_JSONs'

os.makedirs(download_dir, exist_ok=True) # ensure download_dir exists

# load jsons
with open(matching_cities_path, 'r') as f:
    matching_cities = json.load(f)
with open(all_OCM_site_tree_path, 'r') as f:
    all_OCM_site_tree = json.load(f)

def download_file(url, save_path):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            with open(save_path, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=os.path.basename(save_path)
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    bar.update(len(data))
            print(f"downloaded: {save_path}")
        else:
            print(f"dailed to download {url} (status Code: {response.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"error downloading {url}: {e}")

for state_name, cities in all_OCM_site_tree.items():
    for city_name, city_data in cities.items():
        normalized_city_name = city_name.lower().strip()
        
        if normalized_city_name in matching_cities:
            city_fips = matching_cities[normalized_city_name]['fips_code']
            state_fips = matching_cities[normalized_city_name]['state_fips_code']
            
            if city_data["fips_code"] == city_fips:
                print(f"processing city: {city_name} in {state_name}")
                
                json_links = city_data.get("json_links", [])
                
                for link in tqdm(json_links, desc=f"Downloading {city_name}", unit="file"):
                    file_name = f"{state_name}_{city_name}_{os.path.basename(link)}"
                    save_path = os.path.join(download_dir, file_name)
                    
                    if os.path.exists(save_path):
                        print(f"file already exists: {save_path}, skipping download.")
                    else:
                        # Download the file if it doesn't exist
                        download_file(link, save_path)

print("download process complete.")
