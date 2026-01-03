from tokenizer.tokenizer import build_bpe_tokenizer, build_wordlevel_tokenizer
from training.dataset import load_mt_eng_vietnamese_dataset


if __name__ == "__main__":
    dataset = load_mt_eng_vietnamese_dataset(
        split="train",
        seed=24
    )

    src_tokenizer = build_bpe_tokenizer(
        dataset=dataset,
        vocab_size=16000
    )
    tgt_tokenizer = build_wordlevel_tokenizer(
        dataset=dataset,
        vocab_size=12000
    )

    src_tokenizer.save("pretrained/src_tokenizer.json")
    tgt_tokenizer.save("pretrained/tgt_tokenizer.json")