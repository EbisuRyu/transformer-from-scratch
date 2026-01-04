from datasets import load_dataset, Dataset
from pathlib import Path
from typing import Literal


Split = Literal["train", "validation", "test"]


def load_iwslt2015_en_vi(
    split: Split,
    local_dir: str | None = None,
):
    try:
        return load_dataset(
            "mt_eng_vietnamese",
            "iwslt2015-en-vi",
            split=split,
            trust_remote_code=True,
        )

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

        return Dataset.from_list(data)