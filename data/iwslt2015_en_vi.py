from datasets import load_dataset
from pathlib import Path


OUT_DIR = Path("./data/iwslt2015_en_vi")
OUT_DIR.mkdir(parents=True, exist_ok=True)


SPLITS = {
    "train": "train",
    "validation": "validation",
    "test": "test",
}


def export_split(split_name, hf_split):
    print(f"Exporting {split_name}...")
    dataset = load_dataset(
        "mt_eng_vietnamese",
        "iwslt2015-en-vi",
        trust_remote_code=True,
        split=hf_split
    )

    with open(OUT_DIR / f"{split_name}.en", "w", encoding="utf-8") as f_en, \
         open(OUT_DIR / f"{split_name}.vi", "w", encoding="utf-8") as f_vi:
        for ex in dataset:
            en = ex["translation"]["en"].strip()
            vi = ex["translation"]["vi"].strip()

            if en and vi:
                f_en.write(en + "\n")
                f_vi.write(vi + "\n")


if __name__ == "__main__":

    for split_name, hf_split in SPLITS.items():
        export_split(split_name, hf_split)
