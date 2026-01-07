import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tokenizers import Tokenizer

from model.transformer import Transformer
from training.trainer import Trainer
from training.dataloader import get_mt_eng_vietnamese_dataloaders
from training.scheduler import WarmupScheduler
from training.checkpointer import Checkpointer
from utils.logging import get_logger
from config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer from scratch")

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint (last.pt or best.pt)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from training config",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger = get_logger(name="train")
    logger.info(f"Using device: {args.device}")

    logger.info("Loading tokenizers...")
    src_tokenizer = Tokenizer.from_file("./tokenizer/src_tokenizer.json")
    tgt_tokenizer = Tokenizer.from_file("./tokenizer/tgt_tokenizer.json")

    train_config, model_config = load_config()
    train_config.device = args.device
    
    if args.batch_size is not None:
        logger.info(f"Override batch size → {args.batch_size}")
        train_config.batch_size = args.batch_size

    if args.epochs is not None:
        logger.info(f"Override epochs → {args.epochs}")
        train_config.num_epochs = args.epochs
        
    logger.info("Loaded training & model config")

    logger.info("Building dataloaders...")
    dataloaders = get_mt_eng_vietnamese_dataloaders(
        batch_size=train_config.batch_size,
        max_seq_len=model_config.max_seq_len,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
    )

    logger.info("Building Transformer model...")
    model = Transformer(config=model_config).to(args.device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=src_tokenizer.token_to_id("[PAD]"),
        label_smoothing=0.1,
        reduction="mean",
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=train_config.weight_decay,
    )

    scheduler = WarmupScheduler(
        optimizer=optimizer,
        d_model=model_config.d_model,
        warmup_steps=train_config.warmup_steps,
    )

    checkpointer = Checkpointer(
        checkpoint_dir=train_config.checkpoint_dir,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )

    start_epoch = 1

    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")

        model, last_epoch, last_metrics = checkpointer.load(
            name="last" if "last" in args.resume else "best",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=args.device,
        )

        start_epoch = last_epoch + 1
        logger.info(
            f"Resumed training from epoch {last_epoch}, "
            f"val_loss={last_metrics.get('val_loss')}"
        )

    trainer = Trainer(
        config=train_config,
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpointer=checkpointer
    )

    logger.info("Start training...")
    trainer.fit(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["validation"],
        start_epoch=start_epoch,
    )

    logger.success("Training finished successfully!")
    

if __name__ == "__main__":
    main()