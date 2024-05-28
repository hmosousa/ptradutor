import json
import logging

import datasets
from fire import Fire

from src.constants import DATA_PATH
from src.process import huggingface_dataset_filter, valid_n_tokens

logging.basicConfig(level=logging.INFO)


def raw():
    texts = set()
    train, test = [], []
    for filepath in DATA_PATH.glob("*.json"):
        logging.info(f"Formatting {filepath.stem}.")
        content = json.load(filepath.open())

        for id_, info in content.items():
            split = info.pop("split")

            # remove duplicates
            if info["pt"].lower() in texts:
                continue
            texts.add(info["pt"].lower())

            # remove if the text exceed the 1024 tokens
            if valid_n_tokens(
                f"{info['pt']} {info['en']}", min_n_tokens=0
            ):  # leave some margin for extra tokens
                continue

            if split == "test":
                test.append(info)
            else:
                train.append(info)

    raw_ds = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_list(train),
            "test": datasets.Dataset.from_list(test),
        }
    )

    raw_ds.push_to_hub("liaad/PTradutor", "raw")


def clean():
    raw_ds = datasets.load_dataset("liaad/PTradutor", "raw")
    clean_ds = huggingface_dataset_filter(raw_ds)
    clean_ds.push_to_hub("liaad/PTradutor", "clean")


def main(subset: str = "clean"):
    match subset:
        case "raw":
            raw()
        case "clean":
            clean()
        case _:
            raise ValueError(f"Subset {subset} not found.")


if __name__ == "__main__":
    Fire(main)
