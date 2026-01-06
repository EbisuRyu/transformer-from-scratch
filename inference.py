import argparse

import torch
from tokenizers import Tokenizer

from model.transformer import Transformer
from utils.decode import greedy_decode, beam_search_decode
from utils.logging import get_logger
from config import load_config


logger = get_logger(name="translate")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate using Transformer (Greedy / Beam Search)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="greedy",
        choices=["greedy", "beam"],
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--sentence",
        type=str,
        default="hello",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    return parser.parse_args()


def translate(
    model: Transformer,
    src_sentence: str,
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    max_seq_len: int,
    strategy: str = "greedy",
    beam_size: int = 5,
    device: torch.device = torch.device("cpu"),
) -> str:
    model.eval()

    logger.info(f"Translation strategy: {strategy}")
    logger.info(f"Input sentence: {src_sentence}")

    src_ids = src_tokenizer.encode(src_sentence).ids
    src_tokens = torch.tensor(
        [src_ids],
        dtype=torch.long,
        device=device,
    )

    bos_id = tgt_tokenizer.token_to_id("[BOS]")
    eos_id = tgt_tokenizer.token_to_id("[EOS]")

    with torch.no_grad():
        if strategy == "greedy":
            output_ids = greedy_decode(
                model=model,
                src_tokens=src_tokens,
                max_seq_len=max_seq_len,
                bos_token_id=bos_id,
                eos_token_id=eos_id,
                device=device,
            )

        elif strategy == "beam":
            output_ids = beam_search_decode(
                model=model,
                src_tokens=src_tokens,
                max_seq_len=max_seq_len,
                bos_token_id=bos_id,
                eos_token_id=eos_id,
                beam_size=beam_size,
                device=device,
            )

        else:
            raise ValueError("strategy must be 'greedy' or 'beam'")

    translation = tgt_tokenizer.decode(output_ids.tolist())
    logger.success(f"Translation output: {translation}")

    return translation


def main():
    args = parse_args()
    device = torch.device(args.device)

    logger.info("Loading tokenizers...")
    src_tokenizer = Tokenizer.from_file("./tokenizer/src_tokenizer.json")
    tgt_tokenizer = Tokenizer.from_file("./tokenizer/tgt_tokenizer.json")

    logger.info("Building Transformer model...")
    train_config, model_config = load_config()
    model = Transformer(config=model_config).to(device)

    logger.info(f"Running translation on device: {device}")

    translate(
        model=model,
        src_sentence=args.sentence,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_seq_len=args.max_seq_len,
        strategy=args.strategy,
        beam_size=args.beam_size,
        device=device,
    )


if __name__ == "__main__":
    main()
