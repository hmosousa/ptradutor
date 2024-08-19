import json
import logging

import datasets
from fire import Fire
from tqdm import tqdm

from src.constants import DATA_PATH
from src.process import (
    drop_duplicates,
    drop_duplicates_start_ends,
    drop_justext_bad_class,
    huggingface_dataset_filter,
    huggingface_dataset_transform,
)

logging.basicConfig(level=logging.INFO)


def raw():
    train, valid = [], []
    idx = 0
    for filepath in DATA_PATH.glob("*.json"):
        logging.info(f"Formatting {filepath.stem}.")
        content = json.load(filepath.open())
        for _, info in tqdm(content.items()):
            info.pop("split")
            info["idx"] = idx
            idx += 1
            if info["source"] == "dsl_tl":
                valid.append(info)
            else:
                train.append(info)

    logging.debug(f"Train: {len(train)}")
    logging.debug(f"Valid: {len(valid)}")

    logging.info("Pushing to Hugging Face Datasets.")
    dataset = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_list(train),
            "valid": datasets.Dataset.from_list(valid),
        }
    )
    dataset.push_to_hub("u1537782/PTradutor", "raw")


def clean():
    """Push the clean version of the dataset.
    NOTE: It requires that the raw version is already pushed to the hub.
    """
    trainset = datasets.load_dataset("u1537782/PTradutor", name="raw", split="train")
    trainset = drop_duplicates(trainset)
    trainset = drop_duplicates_start_ends(trainset)
    trainset = huggingface_dataset_transform(trainset)
    trainset = huggingface_dataset_filter(trainset)
    validset = datasets.load_dataset("u1537782/PTradutor", name="raw", split="valid")
    dataset = datasets.DatasetDict(
        {
            "train": trainset,
            "valid": validset
        }
    )
    dataset.push_to_hub("u1537782/PTradutor", "clean")


def superclean():
    """Push the superclean version of the dataset.
    NOTE: It requires that the `clean` version is already pushed to the hub.
    """
    trainset = datasets.load_dataset("u1537782/PTradutor", name="clean", split="train")
    trainset = drop_justext_bad_class(trainset)
    validset = datasets.load_dataset("u1537782/PTradutor", name="clean", split="valid")
    dataset = datasets.DatasetDict(
        {
            "train": trainset,
            "valid": validset
        }
    )
    dataset.push_to_hub("u1537782/PTradutor", "superclean")


def main(subset: str = "raw"):
    match subset:
        case "raw":
            raw()
        case "clean":
            clean()
        case "superclean":
            superclean()
        case _:
            raise ValueError(f"Subset {subset} not found.")


if __name__ == "__main__":
    Fire(main)
