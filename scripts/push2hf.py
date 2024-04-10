import json
import logging

import datasets

from src.constants import DATA_PATH

HF_PATH = DATA_PATH / "hf"
HF_PATH.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)

train, test = [], []
for filepath in DATA_PATH.glob("*.json"):
    logging.info(f"Formatting {filepath.stem}.")
    lang = filepath.stem
    content = json.load(filepath.open())
    for id_, info in content.items():
        split = info.pop("split")
        if split == "train":
            train.append(info)
        else:
            test.append(info)

train_path = HF_PATH / "train.jsonl"
json.dump(train, train_path.open("w"))
test_path = HF_PATH / "test.jsonl"
json.dump(test, test_path.open("w"))

data_files = {"train": str(train_path), "test": str(test_path)}
dataset = datasets.load_dataset("json", data_files=data_files)
dataset.push_to_hub("liaad/PTradutor")
