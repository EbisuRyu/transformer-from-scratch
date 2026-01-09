import torch
from tokenizers import Tokenizer

from model.transformer import Transformer
from utils.decode import greedy_decode, beam_search_decode


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

    return translation