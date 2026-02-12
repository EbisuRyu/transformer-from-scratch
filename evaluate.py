import argparse

import torch
from tqdm import tqdm
from tokenizers import Tokenizer

from model.transformer import Transformer
from training.evaluator import TranslationEvaluator
from config import load_config_from_yaml
from utils.logging import get_logger
from utils.translate import translate


logger = get_logger(name="evaluate")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MT model")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model weights",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for model & BERTScore",
    )

    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
    )

    return parser.parse_args()


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    args = parse_args()
    
    logger.info("Loaded training & model config")
    train_config = load_config_from_yaml(
        config_type='train',
        file_path='./configs/train.yaml'
    )
    model_config = load_config_from_yaml(
        config_type='model',
        file_path='./configs/model.yaml'
    )
    
    if args.checkpoint is not None:
        logger.info(f"Override checkpoint → {args.checkpoint}")
        evaluate_config.checkpoint = args.checkpoint
    
    if args.device is not None:
        logger.info(f"Override device → {args.device}")
        evaluate_config.device = args.device
    
    if args.max_seq_len is not None:
        logger.info(f"Override max_seq_len → {args.max_seq_len}")
        evaluate_config.max_seq_len = args.max_seq_len
    
    if args.strategy is not None:
        logger.info(f"Override strategy → {args.strategy}")
        evaluate_config.strategy = args.strategy
    
    if args.beam_size is not None:
        logger.info(f"Override beam_size → {args.beam_size}")
        evaluate_config.beam_size = args.beam_size

    logger.info(f"Using device: {train_config.device}")
    
    logger.info("Loading tokenizers...")
    src_tokenizer = Tokenizer.from_file("./tokenizer/en_tokenizer.json")
    tgt_tokenizer = Tokenizer.from_file("./tokenizer/vi_tokenizer.json")

    model_config = load_config_from_yaml(
        config_type="model",
        file_path="./configs/model.yaml"
    )
    evaluate_config = load_config_from_yaml(
        config_type="evaluate",
        file_path="./configs/evaluate.yaml"
    )
    
    model = Transformer(config=model_config).to(evaluate_config.device)
    
    logger.info(f"Loading checkpoint from {evaluate_config.checkpoint}")
    checkpoint = torch.load(
        evaluate_config.checkpoint,
        map_location=evaluate_config.device,
    )
    model.load_state_dict(
        checkpoint["model_state"],
        strict=False,
    )
    model.eval()

    logger.info(f"Loading source file: {evaluate_config.src_file}")
    sources = read_lines(evaluate_config.src_file)

    logger.info(f"Loading reference file: {evaluate_config.ref_file}")
    references = read_lines(evaluate_config.ref_file)
    
    pairs = [
        (s, r)
        for s, r in zip(sources, references)
        if len(src_tokenizer.encode(s)) <= evaluate_config.max_seq_len
        and len(tgt_tokenizer.encode(r)) <= evaluate_config.max_seq_len
    ]

    sources, references = map(list, zip(*pairs))

    assert len(sources) == len(references), "src/ref line count mismatch"

    logger.info("Translating with model...")
    predictions = []

    for sentence in tqdm(sources, desc="Translating", ncols=90):
        translation = translate(
            model=model,
            src_sentence=sentence,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_seq_len=evaluate_config.max_seq_len,
            strategy=evaluate_config.strategy,
            beam_size=evaluate_config.beam_size,
            device=evaluate_config.device,
        )
        predictions.append(translation)

    evaluator = TranslationEvaluator(
        bert_model=evaluate_config.bert_model,
        device=evaluate_config.device,
    )

    logger.info("Computing metrics (BLEU, chrF++, BERTScore)...")
    scores = evaluator.evaluate(predictions, references)

    logger.info("Evaluation scores")
    for k, v in scores.items():
        logger.info(f"{k:25s}: {v:.4f}")


if __name__ == "__main__":
    main()