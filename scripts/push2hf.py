import json
import logging

import datasets
from transformers import AutoTokenizer

from src.constants import DATA_PATH
from src.utils import contains_html_tags

HF_PATH = DATA_PATH / "hf"
HF_PATH.mkdir(exist_ok=True)

N_CHUNKS = 5

logging.basicConfig(level=logging.INFO)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")


texts = set()  # to keep track of the files that are already in the dataset
train, valid, test = [], [], []
for filepath in DATA_PATH.glob("*.json"):
    logging.info(f"Formatting {filepath.stem}.")
    lang = filepath.stem
    content = json.load(filepath.open())

    for id_, info in content.items():
        split = info.pop("split")

        # remove duplicates
        if info["pt"].lower() in texts:
            continue
        texts.add(info["pt"].lower())

        # remove if the text exceed the 1024 tokens
        tokens = tokenizer(f"{info['pt']} {info['en']}")
        if len(tokens["input_ids"]) > 950:  # leave some margin for extra tokens
            continue

        # remove if the text contains HTML tags
        if contains_html_tags(info["pt"]):
            continue

        if split == "test":
            test.append(info)
        elif len(valid) < 256 and info["source"] == "dsl_tl":
            valid.append(info)
        else:
            train.append(info)

print(valid)

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

valid_path = HF_PATH / "valid.jsonl"
json.dump(valid, valid_path.open("w"))

test_path = HF_PATH / "test.jsonl"
json.dump(test, test_path.open("w"))

data_files = {"train": train_fps, "valid": str(valid_path), "test": str(test_path)}
dataset = datasets.load_dataset("json", data_files=data_files)
dataset.push_to_hub("liaad/PTradutor")
