from typing import Optional

import html
import torch
import random
from tokenizers import Tokenizer
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler


def load_mt_eng_vietnamese_dataset(
    split: str = "train",
    seed: int = 24
):
    dataset = load_dataset(
        "mt_eng_vietnamese",
        "iwslt2015-en-vi",
        split=split,
        trust_remote_code=True
    )

    if split == "train":
        dataset = dataset.shuffle(seed=seed)

    def convert_format(example):
        return {
            "src": html.unescape(example["translation"]["en"]),
            "tgt": html.unescape(example["translation"]["vi"]),
        }

    dataset = dataset.map(
        convert_format,
        remove_columns=dataset.column_names
    )
    
    return dataset


def tokenize_and_filter(
    dataset: Dataset,
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    max_seq_len: int
) -> Dataset:

    def tokenize(example):
        src_ids = src_tokenizer.encode(example["src"]).ids
        tgt_ids = tgt_tokenizer.encode(example["tgt"]).ids

        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
            "src_len": len(src_ids),
            "tgt_len": len(tgt_ids),
        }

    dataset = dataset.map(
        tokenize,
        remove_columns=dataset.column_names
    )

    dataset = dataset.filter(
        lambda x: x["src_len"] <= max_seq_len and x["tgt_len"] <= max_seq_len
    )
    dataset = dataset.sort('src_len', reverse = True)
    
    return dataset


class ShuffledBatchSampler(Sampler):
    
    def __init__(self, dataset: Dataset, batch_size: int):
        self.batch_size = batch_size
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        random.shuffle(self.indices)
        for i in range(0, len(self.indices), self.batch_size):
            yield self.indices[i:i+self.batch_size]

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size
    

def get_mt_eng_vietnamese_dataloaders(
    batch_size: int,
    max_seq_len: int,
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    splits: list = ["train", "validation", "test"]
) -> dict:
    dataloaders = {}

    def collate_fn(batch):
        src_pad = src_tokenizer.token_to_id("[PAD]")
        tgt_pad = tgt_tokenizer.token_to_id("[PAD]")
        src = pad_sequence(
            [torch.tensor(b["src_ids"]) for b in batch], 
            batch_first=True, padding_value=src_pad
        )
        tgt = pad_sequence(
            [torch.tensor(b["tgt_ids"]) for b in batch], 
            batch_first=True, padding_value=tgt_pad
        )
        return src, tgt

    for split in splits:
        dataset = load_mt_eng_vietnamese_dataset(split=split)
        dataset = tokenize_and_filter(
            dataset=dataset,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_seq_len=max_seq_len
        )

        batch_sampler = ShuffledBatchSampler(dataset=dataset, batch_size=batch_size)
        dataloaders[split] = DataLoader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_sampler=batch_sampler,
            pin_memory=True
        )

    return dataloaders