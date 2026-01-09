import argparse
import torch
from tokenizers import Tokenizer
from tqdm import tqdm

from model.transformer import Transformer
from training.evaluator import TranslationEvaluator
from config import load_config
from utils.logging import get_logger
from utils.translate import translate


logger = get_logger(name="evaluate")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MT model")

    parser.add_argument(
        "--src-file",
        type=str,
        default="./data/iwslt2015_en_vi/test.en",
        help="Path to source sentences (English)",
    )

    parser.add_argument(
        "--ref-file",
        type=str,
        default="./data/iwslt2015_en_vi/test.vi",
        help="Path to reference translations (Vietnamese)",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        default=None,
        help="Path to trained model weights",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for model & BERTScore",
    )

    parser.add_argument(
        "--bert-model",
        type=str,
        default="xlm-roberta-base",
        help="BERTScore model name",
    )

    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="greedy",
        choices=["greedy", "beam"],
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
    )

    return parser.parse_args()


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    args = parse_args()

    logger.info(f"Using device: {args.device}")

    logger.info("Loading tokenizers...")
    src_tokenizer = Tokenizer.from_file("./tokenizer/src_tokenizer.json")
    tgt_tokenizer = Tokenizer.from_file("./tokenizer/tgt_tokenizer.json")

    train_config, model_config = load_config()
    model = Transformer(config=model_config).to(args.device)

    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(
        args.checkpoint,
        map_location=args.device,
    )
    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )
    model.eval()

    logger.info(f"Loading source file: {args.src_file}")
    sources = read_lines(args.src_file)[:2]

    logger.info(f"Loading reference file: {args.ref_file}")
    references = read_lines(args.ref_file)[:2]

    assert len(sources) == len(references), "src/ref line count mismatch"

    logger.info("Translating with model...")
    predictions = []

    for s in tqdm(sources, desc="Translating", ncols=90):
        vi = translate(
            model=model,
            src_sentence=s,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_seq_len=args.max_seq_len,
            strategy=args.strategy,
            beam_size=args.beam_size,
            device=args.device,
        )
        predictions.append(vi)

    evaluator = TranslationEvaluator(
        bert_model=args.bert_model,
        device=args.device,
    )

    logger.info("Computing metrics (BLEU, chrF++, BERTScore)...")
    scores = evaluator.evaluate(predictions, references)

    logger.info("Evaluation scores")
    for k, v in scores.items():
        logger.info(f"{k:25s}: {v:.4f}")


if __name__ == "__main__":
    main()