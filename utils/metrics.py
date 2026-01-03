from typing import List

import numpy as np
from tokenizers import Tokenizer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def compute_bleu(
    references: List[List[List[str]]],  # list of list of reference sentences (tokenized)
    predictions: List[List[str]]        # list of predicted sentences (tokenized)
) -> float:
    """
    Compute corpus BLEU score (nltk)
    """
    smoothing = SmoothingFunction().method4
    bleu = corpus_bleu(references, predictions, smoothing_function=smoothing)
    return bleu


def compute_bleu_from_ids(
    pred_ids: np.ndarray,
    tgt_ids: np.ndarray,
    tokenizer: Tokenizer
) -> float:
    """
    Decode pred_ids and tgt_ids using tokenizer, then compute BLEU.
    pred_ids: (batch_size, seq_len)
    tgt_ids: (batch_size, seq_len)
    """
    pred_texts = tokenizer.decode_batch(pred_ids, skip_special_tokens=True)
    tgt_texts = tokenizer.decode_batch(tgt_ids, skip_special_tokens=True)

    predictions = [p.split() for p in pred_texts]
    references = [[t.split()] for t in tgt_texts]

    return compute_bleu(references, predictions)


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss
    """
    return np.exp(loss)
