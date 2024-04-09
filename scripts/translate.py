import logging

from fire import Fire
from tqdm import tqdm

from src.data import TranslationDataset, load_dataset
from src.translator import Translator


logging.basicConfig(level=logging.INFO)

def main(lang="en", name="pt_vid", domain="default", split="train"):
    """
    name = ["pt_vid", "frmt", "dsl_tl"]

    Run the following commands to translate the datasets:

    python scripts/translate.py -l "en" -n "pt_vid" -d "journalistic" -s "train"
    python scripts/translate.py -l "en" -n "pt_vid" -d "legal" -s "train"
    python scripts/translate.py -l "en" -n "pt_vid" -d "literature" -s "train"
    python scripts/translate.py -l "en" -n "pt_vid" -d "politics" -s "train"
    python scripts/translate.py -l "en" -n "pt_vid" -d "social_media" -s "train"
    python scripts/translate.py -l "en" -n "pt_vid" -d "web" -s "train"
    python scripts/translate.py -l "en" -n "dsl_tl" -d "default" -s "train"
    python scripts/translate.py -l "en" -n "dsl_tl" -d "default" -s "test"
    python scripts/translate.py -l "en" -n "frmt" -d "default" -s "train"
    python scripts/translate.py -l "en" -n "frmt" -d "default" -s "test"
    """
    translator = Translator(source="pt", target=lang)

    idx = 0
    
    logging.info(f"Translating dataset {name}.")
    dataset = load_dataset(name)
    source_ds = TranslationDataset(f"{lang}_{name}_{domain}_{split}")
    for text in tqdm(dataset[domain][split]):
        if idx not in source_ds:
            try:
                translate = translator.translate(text)
                data = {
                    "idx": idx,
                    "source": name,
                    "domain": domain,
                    "split": split,
                    "pt": text, 
                    "en": translate
                }
                source_ds.add(idx, data)
            except Exception as e:
                print(f"Error translating {idx}: {e}")
        else:
            logging.info(f"Skipping {idx}.")
        idx += 1
        if idx % 100 == 0:
            source_ds.save()
    source_ds.save()


if __name__ == "__main__":
    Fire(main)
