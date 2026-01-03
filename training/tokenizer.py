from typing import Optional
from torch.utils.data import Dataset

from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE, WordLevel
from tokenizers.normalizers import Lowercase, NFD
from tokenizers.pre_tokenizers import Whitespace, WhitespaceSplit
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from tokenizers.processors import TemplateProcessing


def build_bpe_tokenizer(
    dataset: Dataset, 
    vocab_size: int
) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    tokenizer.normalizer = normalizers.Sequence([
        NFD(),
        Lowercase()
    ])

    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        show_progress=True
    )

    def batch_iterator(batch_size=10000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i: i + batch_size]["src"]  # EN ONLY

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )

    return tokenizer


def build_wordlevel_tokenizer(
    dataset: Dataset, 
    vocab_size: int
) -> Tokenizer:
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))

    tokenizer.normalizer = normalizers.Sequence([
        NFD(),
        Lowercase()
    ])

    tokenizer.pre_tokenizer = WhitespaceSplit()

    trainer = WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        show_progress=True
    )

    def batch_iterator(batch_size=10000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i: i + batch_size]["tgt"] 

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )

    return tokenizer