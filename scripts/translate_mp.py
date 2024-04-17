import concurrent.futures
import logging
import time

from fire import Fire
from tqdm import tqdm

from src.data import TranslationDataset, load_dataset
from src.translator import Translator

logging.basicConfig(level=logging.INFO)


def main(lang="en", name="pt_vid", domain="journalistic", split="train"):
    """
    name = ["pt_vid", "frmt", "dsl_tl"]

    Run the following commands to translate the datasets:

    python scripts/translate_mp.py -l "en" -n "pt_vid" -d "journalistic" -s "train"
    python scripts/translate_mp.py -l "en" -n "pt_vid" -d "legal" -s "train"
    python scripts/translate_mp.py -l "en" -n "pt_vid" -d "literature" -s "train"
    python scripts/translate_mp.py -l "en" -n "pt_vid" -d "politics" -s "train"
    python scripts/translate_mp.py -l "en" -n "pt_vid" -d "social_media" -s "train"
    python scripts/translate_mp.py -l "en" -n "pt_vid" -d "web" -s "train"
    python scripts/translate_mp.py -l "en" -n "dsl_tl" -d "default" -s "train"
    python scripts/translate_mp.py -l "en" -n "dsl_tl" -d "default" -s "test"
    python scripts/translate_mp.py -l "en" -n "frmt" -d "default" -s "train"
    python scripts/translate_mp.py -l "en" -n "frmt" -d "default" -s "test"
    """
    translator = Translator(source="pt", target=lang)

    def translate(text):
        try:
            translate = translator(text)
            time.sleep(3)
            return translate
        except Exception as e:
            logging.error(f"Error translating {text}: {e}")

    logging.info(f"Translating dataset {name}.")
    dataset = load_dataset(name)
    translation_ds = TranslationDataset(f"{lang}_{name}_{domain}_{split}_temp")

    logging.info("Filtering missing translations.")
    texts = dataset[domain][split]
    ids = list(range(len(dataset[domain][split])))
    missing_ids = list(set(ids) - set(map(int, translation_ds.ids)))
    texts = [texts[idx] for idx in missing_ids]

    logging.info("Batching the data.")
    bids = [missing_ids[i : i + 100] for i in range(0, len(missing_ids), 100)]
    btexts = [texts[i : i + 100] for i in range(0, len(texts), 100)]

    logging.info("Translating texts.")
    for bid, btext in zip(bids, btexts):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(tqdm(executor.map(translate, btext), total=len(btext)))

            logging.info("Saving translations.")
            for idx, text, result in zip(bid, btext, results):
                data = {
                    "idx": idx,
                    "source": name,
                    "domain": domain,
                    "split": split,
                    "pt": text,
                    "en": result,
                }
                translation_ds.add(idx, data)
            translation_ds.save()


if __name__ == "__main__":
    Fire(main)
