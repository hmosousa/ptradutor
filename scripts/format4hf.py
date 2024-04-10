import json
import logging

from src.constants import DATA_PATH

HF_PATH = DATA_PATH / "hf" / "PTradutor"
HF_PATH.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)


for filepath in DATA_PATH.glob("*.json"):
    logging.info(f"Formatting {filepath.stem}.")
    lang = filepath.stem
    content = json.load(filepath.open())

    train, test = [], []
    for id_, info in content.items():
        split = info.pop("split")
        if split == "train":
            train.append(info)
        else:
            test.append(info)
    
    json.dump(train, (HF_PATH / "train.jsonl").open("w"))
    json.dump(test, (HF_PATH / "test.jsonl").open("w"))
