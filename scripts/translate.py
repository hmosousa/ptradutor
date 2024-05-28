import logging

from fire import Fire
from tqdm import tqdm

from src.data import TranslationDataset, load_dataset
from src.translator import Translator

logging.basicConfig(level=logging.INFO)


def main(lang="en", name="dsl_tl", domain="default", split="train"):
    """
    name = ["pt_vid", "dsl_tl"]

    Run the following commands to translate the datasets:

    python scripts/translate.py -l "en" -n "pt_vid" -d "journalistic" -s "train"
    python scripts/translate.py -l "en" -n "pt_vid" -d "legal" -s "train"
    python scripts/translate.py -l "en" -n "pt_vid" -d "literature" -s "train"
    python scripts/translate.py -l "en" -n "pt_vid" -d "politics" -s "train"
    python scripts/translate.py -l "en" -n "pt_vid" -d "social_media" -s "train"
    python scripts/translate.py -l "en" -n "pt_vid" -d "web" -s "train"
    python scripts/translate.py -l "en" -n "dsl_tl" -d "default" -s "train"
    python scripts/translate.py -l "en" -n "dsl_tl" -d "default" -s "test"
    """
    translator = Translator(source="pt", target=lang)

    logging.info(f"Loading dataset {name}.")
    dataset = load_dataset(name)
    texts = dataset[domain][split]

    ds_name = f"{lang}_{name}_{domain}_{split}"
    logging.info(f"Creating dataset {ds_name}. (loading in case it exists)")
    translation_ds = TranslationDataset(ds_name)

    logging.info(f"Translating dataset {name}.")
    for idx, text in tqdm(enumerate(texts)):
        if idx not in translation_ds:
            try:
                translate = translator.translate(text)
                data = {
                    "idx": idx,
                    "source": name,
                    "domain": domain,
                    "split": split,
                    "pt": text,
                    "en": translate,
                }
                translation_ds.add(idx, data)
            except Exception as e:
                print(f"Error translating {idx}: {e}")
        else:
            logging.info(f"Skipping {idx}.")

        if idx % 100 == 0:
            logging.info(f"Saving dataset {ds_name}.")
            translation_ds.save()

    translation_ds.save()


if __name__ == "__main__":
    Fire(main)
