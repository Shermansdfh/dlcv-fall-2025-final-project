import json
import os
import argparse

def update_image_path(json_dir, new_img_dir, output_dir=None):
    if output_dir is None:
        output_dir = json_dir

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(json_dir):
        if not filename.endswith(".json"):
            continue

        json_path = os.path.join(json_dir, filename)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 取得原本圖片檔名
        old_image_path = data.get("image_path", "")
        image_name = os.path.basename(old_image_path)

        # 組合新的 image_path
        data["image_path"] = os.path.join(new_img_dir, image_name)

        # 輸出（可覆蓋或存到新資料夾）
        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Updated: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", required=True, help="JSON 資料夾路徑")
    parser.add_argument("--img_dir", required=True, help="新的 image folder 路徑")
    parser.add_argument("--output_dir", default=None, help="輸出資料夾（預設覆蓋原檔）")

    args = parser.parse_args()

    update_image_path(args.json_dir, args.img_dir, args.output_dir)
