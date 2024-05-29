import json
import logging

import datasets
from fire import Fire
from tqdm import tqdm

from src.constants import DATA_PATH
from src.process import huggingface_dataset_filter, huggingface_dataset_transform, valid_n_tokens, MAX_N_TOKENS

logging.basicConfig(level=logging.INFO)


def raw():
    texts = set()
    train, test = [], []
    for filepath in DATA_PATH.glob("*.json"):
        logging.info(f"Formatting {filepath.stem}.")
        content = json.load(filepath.open())

        for _, info in tqdm(content.items()):
            split = info.pop("split")

            # remove duplicates
            if info["pt"].lower() in texts:
                continue
            texts.add(info["pt"].lower())
                
            # remove if the text exceeds MAX_N_TOKENS
            prompt = f"{info['pt']} {info['en']}"
            if len(prompt) > MAX_N_TOKENS:  # to make it run faster     
                if not valid_n_tokens(
                    f"{info['pt']} {info['en']}", min_n_tokens=0
                ):  # leave some margin for extra tokens
                    continue

            if split == "test":
                test.append(info)
            else:
                train.append(info)

    logging.debug(f"Train: {len(train)}")
    logging.debug(f"Test: {len(test)}")

    logging.info("Pushing to Hugging Face Datasets.")
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
    clean_ds = huggingface_dataset_transform(clean_ds)
    clean_ds.push_to_hub("liaad/PTradutor", "clean")


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
