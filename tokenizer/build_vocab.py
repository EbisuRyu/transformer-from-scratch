from tokenizer.tokenizer import build_bpe_tokenizer, build_wordlevel_tokenizer
from utils.dataset import load_iwslt2015_en_vi


if __name__ == "__main__":
    dataset = load_iwslt2015_en_vi(
        split="train",
        local_dir="./data/load_iwslt2015_en_vi",
    )

    src_tokenizer = build_wordlevel_tokenizer(
        dataset=dataset,
        vocab_size=16000,
        fields=["en"]
    )
    tgt_tokenizer = build_wordlevel_tokenizer(
        dataset=dataset,
        vocab_size=12000,
        fields=["vi"]
    )

    src_tokenizer.save("tokenizer/en_tokenizer.json")
    tgt_tokenizer.save("tokenizer/vi_tokenizer.json")