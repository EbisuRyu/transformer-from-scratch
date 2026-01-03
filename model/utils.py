import torch


def make_pad_mask(seq: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    seq: (batch, seq_len)
    return: (batch, 1, 1, seq_len)
    """
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    return: (1, 1, seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)
