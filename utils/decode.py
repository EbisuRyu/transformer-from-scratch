import argparse

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model.transformer import Transformer


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src_tokens: torch.Tensor,
    max_seq_len: int,
    bos_token_id: int,
    eos_token_id: int,
    device: torch.device,
):
    model.eval()

    # target sequence starts with <BOS>
    ys = torch.tensor([[bos_token_id]], device=device)

    for _ in range(max_seq_len - 1):
        
        outputs = model(
            src=src_tokens,
            tgt=ys
        )  
        logits = outputs["logits"]

        next_token = logits[:, -1, :].argmax(dim=-1).item()

        ys = torch.cat(
            [ys, torch.tensor([[next_token]], device=device)],
            dim=1,
        )

        if next_token == eos_token_id:
            break

    return ys.squeeze(0)


@torch.no_grad()
def beam_search_decode(
    model: Transformer,
    src_tokens: torch.Tensor,
    max_seq_len: int,
    bos_token_id: int,
    eos_token_id: int,
    beam_size: int,
    device: torch.device,
):
    model.eval()

    # (token_sequence, log_prob)
    sequences = [
        (torch.tensor([bos_token_id], device=device), 0.0)
    ]

    for _ in range(max_seq_len - 1):
        all_candidates = []

        for seq, score in sequences:
            if seq[-1].item() == eos_token_id:
                all_candidates.append((seq, score))
                continue

            tgt = seq.unsqueeze(0)  # (1, T)
            outputs = model(
                src=src_tokens,
                tgt=tgt
            )
            logits = outputs["logits"]

            log_probs = F.log_softmax(
                logits[:, -1, :], dim=-1
            )

            topk_log_probs, topk_tokens = log_probs.topk(
                beam_size
            )

            for k in range(beam_size):
                candidate_seq = torch.cat(
                    [seq, topk_tokens[0, k].unsqueeze(0)]
                )
                candidate_score = (
                    score + topk_log_probs[0, k].item()
                )
                all_candidates.append(
                    (candidate_seq, candidate_score)
                )

        sequences = sorted(
            all_candidates,
            key=lambda x: x[1],
            reverse=True,
        )[:beam_size]

        if all(
            seq[-1].item() == eos_token_id
            for seq, _ in sequences
        ):
            break

    return sequences[0][0]