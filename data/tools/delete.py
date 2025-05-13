import copy
import json
from pathlib import Path


dir_path = Path('data/raw/vt-dex-manip/info')
json_files = dir_path.glob('*.json')

for file in json_files:

    with open(file, "r") as f:
        data = json.load(f)

    new_item = dict()
    for i, item in enumerate(data):
        new_item["id"] = item["id"]
        new_item["class"] = item["class"]
        new_item["class_id"] = item["class_id"]
        data[i] = copy.deepcopy(new_item)

    with open(file, "w") as f:
        json.dump(data, f, indent=4)
