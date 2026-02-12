import argparse

import torch
from tokenizers import Tokenizer

from config import load_config_from_yaml
from model.transformer import Transformer
from utils.logging import get_logger
from utils.translate import translate


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
        default=40,
    )
    parser.add_argument(
        "--sentence",
        type=str,
        default="you can see",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./weights/best.pt"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    logger.info("Loading tokenizers...")
    src_tokenizer = Tokenizer.from_file("./tokenizer/en_tokenizer.json")
    tgt_tokenizer = Tokenizer.from_file("./tokenizer/vi_tokenizer.json")

    logger.info("Building Transformer model...")
    model_config = load_config_from_yaml(
        config_type='model',
        file_path='./configs/model.yaml'
    )
    model = Transformer(config=model_config).to(device)
    
    logger.info(f"Using checkpoint: {args.checkpoint}")
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )
    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )
    
    logger.info(f"Running translation on device: {device}")

    logger.info(f"Translation strategy: {args.strategy}")
    logger.info(f"Input sentence: {args.sentence}")
    
    translation = translate(
        model=model,
        src_sentence=args.sentence,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_seq_len=args.max_seq_len,
        strategy=args.strategy,
        beam_size=args.beam_size,
        device=device,
    )
    
    logger.success(f"Translation output: {translation}")


if __name__ == "__main__":
    main()