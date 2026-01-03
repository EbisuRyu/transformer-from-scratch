from typing import List, Dict
import html
import random
import torch
from tokenizers import Tokenizer
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Sampler


class TranslationDataset(Dataset):
    def __init__(self, src_texts: List[str], tgt_texts: List[str], src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer):
        assert len(src_texts) == len(tgt_texts)
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_ids = self.src_tokenizer.encode(self.src_texts[idx]).ids
        tgt_ids = self.tgt_tokenizer.encode(self.tgt_texts[idx]).ids
        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long)
        }


class ShuffledBatchSampler(Sampler):
    def __init__(self, dataset: Dataset, batch_size: int):
        self.dataset_len = len(dataset)
        self.batch_size = batch_size
        self.indices = list(range(self.dataset_len))

    def __iter__(self):
        random.shuffle(self.indices)
        for i in range(0, self.dataset_len, self.batch_size):
            yield self.indices[i:i + self.batch_size]

    def __len__(self):
        return (self.dataset_len + self.batch_size - 1) // self.batch_size


def prepare_dataloader(
    split: str,
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    batch_size: int,
    max_seq_len: int,
    seed: int = 24
) -> DataLoader:
    dataset = load_dataset(
        "mt_eng_vietnamese",
        "iwslt2015-en-vi",
        split=split,
        trust_remote_code=True
    )
    if split == "train":
        dataset = dataset.shuffle(seed=seed)

    src_texts = []
    tgt_texts = []

    for example in dataset:
        src = html.unescape(example["translation"]["en"])
        tgt = html.unescape(example["translation"]["vi"])
        src_ids = src_tokenizer.encode(src).ids
        tgt_ids = tgt_tokenizer.encode(tgt).ids
        if len(src_ids) <= max_seq_len and len(tgt_ids) <= max_seq_len:
            src_texts.append(src)
            tgt_texts.append(tgt)

    translation_dataset = TranslationDataset(src_texts, tgt_texts, src_tokenizer, tgt_tokenizer)

    def collate_fn(batch):
        src_pad = src_tokenizer.token_to_id("[PAD]")
        tgt_pad = tgt_tokenizer.token_to_id("[PAD]")
        src_batch = pad_sequence([b["src_ids"] for b in batch], batch_first=True, padding_value=src_pad)
        tgt_batch = pad_sequence([b["tgt_ids"] for b in batch], batch_first=True, padding_value=tgt_pad)
        return src_batch, tgt_batch

    batch_sampler = ShuffledBatchSampler(translation_dataset, batch_size=batch_size)
    dataloader = DataLoader(translation_dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, pin_memory=True)
    return dataloader


def get_mt_eng_vietnamese_dataloaders(
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    batch_size: int,
    max_seq_len: int,
    splits: List[str] = ["train", "validation", "test"]
) -> Dict[str, DataLoader]:
    return {
        split: prepare_dataloader(split, src_tokenizer, tgt_tokenizer, batch_size, max_seq_len) 
        for split in splits
    }