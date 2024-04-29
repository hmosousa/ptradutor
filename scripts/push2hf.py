import json
import logging

import datasets

from src.constants import DATA_PATH

HF_PATH = DATA_PATH / "hf"
HF_PATH.mkdir(exist_ok=True)

N_CHUNKS = 5

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
n_train = len(train)
logging.debug(f"Train size: {n_train}")
train_fps = []
for chunk in range(N_CHUNKS):
    start = chunk * n_train // N_CHUNKS
    end = (chunk + 1) * n_train // N_CHUNKS
    logging.debug(f"Chunk {chunk}: {start} - {end}")
    chunk_path = HF_PATH / f"train_{chunk}.jsonl"
    json.dump(train[start:end], chunk_path.open("w"))
    train_fps.append(str(chunk_path))

test_path = HF_PATH / "test.jsonl"
json.dump(test, test_path.open("w"))

data_files = {"train": train_fps, "test": str(test_path)}
dataset = datasets.load_dataset("json", data_files=data_files)
dataset.push_to_hub("liaad/PTradutor")
