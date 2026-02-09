from typing import Literal
import re
import unicodedata
from pathlib import Path
from datasets import load_dataset, Dataset


VIETNAMESE_CHARS = (
    "aăâbcdđeêghiklmnoôơpqrstuưvxy"
    "áàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệ"
    "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữự"
    "ýỳỷỹỵ"
)
VI_REGEX = re.compile(rf"[^{VIETNAMESE_CHARS}\s]")


def normalize_vietnamese(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFC", text)
    text = VI_REGEX.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_english(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def process_dataset(dataset: Dataset) -> Dataset:

    def _process(example):
        en_raw = example["translation"]["en"]
        vi_raw = example["translation"]["vi"]

        en_norm = normalize_english(en_raw)
        vi_norm = normalize_vietnamese(vi_raw)

        return {
            "translation": {
                "en": en_norm,
                "vi": vi_norm,
            }
        }

    return dataset.map(_process)


def load_iwslt2015_en_vi_dataset(
    split: Literal["train", "validation", "test"],
    local_dir: str | None = None,
) -> Dataset:
    try:
        dataset = load_dataset(
            "mt_eng_vietnamese",
            "iwslt2015-en-vi",
            split=split,
            trust_remote_code=True,
        )
        dataset = process_dataset(dataset)
        return dataset

    except Exception as e:
        if local_dir is None:
            raise RuntimeError("HF load failed and no local_dir provided") from e

        data_dir = Path(local_dir)
        en_path = data_dir / f"{split}.en"
        vi_path = data_dir / f"{split}.vi"

        if not en_path.exists() or not vi_path.exists():
            raise FileNotFoundError(f"Missing {split}.en / {split}.vi")

        with open(en_path, encoding="utf-8") as f_en, \
             open(vi_path, encoding="utf-8") as f_vi:
            en_lines = f_en.readlines()
            vi_lines = f_vi.readlines()

        if len(en_lines) != len(vi_lines):
            raise ValueError("EN / VI line count mismatch")

        data = [
            {
                "translation": {
                    "en": en.strip(),
                    "vi": vi.strip(),
                }
            }
            for en, vi in zip(en_lines, vi_lines)
            if en.strip() and vi.strip()
        ]

        dataset = Dataset.from_list(data)
        dataset = process_dataset(dataset)
        return dataset