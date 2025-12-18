import os
import json

# 要處理的資料夾
FOLDER_PATH = "outputs/pipeline_outputs/cld"

OLD_PREFIX = "/scratch/comme502/ce505203/finals/inputs/"
NEW_PREFIX = "/workspace/finals/data/inputs/"

for filename in os.listdir(FOLDER_PATH):
    if not filename.endswith(".json"):
        continue

    file_path = os.path.join(FOLDER_PATH, filename)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "image_path" in data:
        # 只取檔名
        image_name = os.path.basename(data["image_path"])
        data["image_path"] = NEW_PREFIX + image_name

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Updated: {filename}")
    else:
        print(f"Skipped (no image_path): {filename}")
