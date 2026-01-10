from typing import Optional
from IPython.display import display, clear_output

import pandas as pd
from tqdm import tqdm
from tokenizers import Tokenizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import TrainConfig
from model.transformer import Transformer
from training.checkpointer import Checkpointer


class Trainer:
    
    def __init__(
        self,
        config: TrainConfig,
        model: Transformer,
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpointer: Checkpointer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        self.config = config
        self.model = model.to(config.device)
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpointer = checkpointer
        self.device = config.device
        
        self.global_step = 0        
        self.history = []

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        batch_iterator = tqdm(
            train_loader,
            desc=f"Training Epoch {epoch:02d}",
            total=len(train_loader)
        )

        for batch in batch_iterator:
            src_batch = batch[0].to(self.device)
            tgt_batch = batch[1].to(self.device)

            decoder_input = tgt_batch[:, :-1]
            decoder_target = tgt_batch[:, 1:]

            outputs = self.model(src_batch, decoder_input)
            logits = outputs["logits"]

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                decoder_target.reshape(-1)
            )
            loss.backward()
            total_loss += loss.item()
            
            self.global_step += 1

            if self.global_step % self.config.grad_accumulation_steps == 0:
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )

                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                self.optimizer.zero_grad()

            if self.scheduler:
                lr = self.scheduler.get_last_lr()[0]
            else:
                lr = self.optimizer.param_groups[0]["lr"]

            batch_iterator.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{lr:.6e}"
            })

        avg_loss = total_loss / len(train_loader)
        return avg_loss


    @torch.no_grad()
    def eval_epoch(self, val_loader: DataLoader, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0

        batch_iterator = tqdm(val_loader, desc=f"Validation Epoch {epoch:02d}", total=len(val_loader))
        for batch in batch_iterator:
            src_batch = batch[0].to(self.device)
            tgt_batch = batch[1].to(self.device)

            decoder_input = tgt_batch[:, :-1]
            decoder_target = tgt_batch[:, 1:]
            
            outputs = self.model(src_batch, decoder_input)
            logits = outputs["logits"]
            
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                decoder_target.reshape(-1)
            )
            total_loss += loss.item()
        
            batch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(val_loader)
        return avg_loss

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        start_epoch: int = 1,
        save_every: int = 3
    ):
        for epoch in range(start_epoch, self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.eval_epoch(val_loader, epoch)

            metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss
            }
            
            if epoch % save_every == 0:
                self.checkpointer.save(
                    epoch=epoch,
                    metrics=metrics,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler
                )

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss
            }
            self.history.append(row)

            df = pd.DataFrame(self.history)

            clear_output(wait=True)
            display(
                df.style.format({
                    "train_loss": "{:.4f}",
                    "val_loss": "{:.4f}"
                })
            )

        return df
