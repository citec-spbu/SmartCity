import os
import shutil

base_dir = "AI_learning_data"
combined_dir = os.path.join(base_dir, "downloaded_JSONs", "filtered", "combined")
target_dir = os.path.join(base_dir, "city_combined_data")
downloaded_jsons_dir = os.path.join(base_dir, "downloaded_JSONs")

os.makedirs(target_dir, exist_ok=True)

for file in os.listdir(combined_dir):
    if file.endswith("COMBINED.json"):
        src_path = os.path.join(combined_dir, file)
        dst_path = os.path.join(target_dir, file)
        shutil.move(src_path, dst_path)
        print(f"moved: {file}")

if os.path.exists(downloaded_jsons_dir):
    shutil.rmtree(downloaded_jsons_dir)
    print(f"deleted folder: {downloaded_jsons_dir}")

print("cleanup complete. all temporary files removed and combined data stored in", target_dir)