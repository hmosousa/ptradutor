import json
import logging

import datasets
from fire import Fire
from tqdm import tqdm

from src.constants import DATA_PATH
from src.process import (
    drop_duplicates,
    drop_duplicates_start_ends,
    huggingface_dataset_filter,
    huggingface_dataset_transform,
)

logging.basicConfig(level=logging.INFO)


def raw():
    train, test = [], []
    idx = 0
    for filepath in DATA_PATH.glob("*.json"):
        logging.info(f"Formatting {filepath.stem}.")
        content = json.load(filepath.open())
        for _, info in tqdm(content.items()):
            split = info.pop("split")
            info["idx"] = idx
            idx += 1
            if split == "test":
                test.append(info)
            else:
                train.append(info)

    logging.debug(f"Train: {len(train)}")
    logging.debug(f"Test: {len(test)}")

    logging.info("Pushing to Hugging Face Datasets.")
    dataset = datasets.DatasetDict({
            "train": datasets.Dataset.from_list(train),
            "test": datasets.Dataset.from_list(test),
    })
    dataset.push_to_hub("liaad/PTradutor", "raw")


def clean():
    """Push the clean version of the dataset.
    NOTE: It requires that the raw version is already pushed to the hub.
    """
    dataset = datasets.load_dataset("liaad/PTradutor", "raw")
    dataset = drop_duplicates(dataset)
    dataset = drop_duplicates_start_ends(dataset)
    dataset = huggingface_dataset_transform(dataset)
    dataset = huggingface_dataset_filter(dataset)
    dataset.push_to_hub("liaad/PTradutor", "clean")


def main(subset: str = "raw"):
    match subset:
        case "raw":
            raw()
        case "clean":
            clean()
        case _:
            raise ValueError(f"Subset {subset} not found.")


if __name__ == "__main__":
    Fire(main)
