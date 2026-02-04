# Attention Is All You Need

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-red?logo=pytorch)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Tokenizers-yellow?logo=huggingface)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A clean PyTorch implementation of the Transformer architecture from the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) for Neural Machine Translation (English → Vietnamese).

<p align="center">
  <img src="public/architecture.png" alt="Transformer Architecture" width="70%">
</p>

## Highlights

This section summarizes the key ideas and strengths of the project. It is a quick snapshot of what the repository provides and why it is useful. Use it to decide if the project fits your goals.

- Modular codebase with clear separation between model, training, and utilities.
- YAML configuration for fast experiment iteration.
- WordLevel tokenizers (English and Vietnamese) built with Hugging Face Tokenizers.
- Ready-to-run scripts for training, inference, and evaluation.

## Repository Structure

This section explains how the repository is organized. It helps you locate code, configs, data, and scripts quickly. Skim this first if you are new to the project.

```
transformer-from-scratch/
├── config/              # Model & training configurations (YAML)
├── data/                # IWSLT2015 EN-VI dataset
├── model/               # Transformer architecture (encoder, decoder, layers)
├── tokenizer/           # WordLevel tokenizers (EN & VI)
├── training/            # Trainer, evaluator, scheduler, checkpointer
├── utils/               # Masks, translation, logging utilities
├── weights/             # Saved checkpoints
├── train.py             # Training entrypoint
├── inference.py         # Translation inference
└── evaluate.py          # Model evaluation
```

## Getting Started

This section walks you through preparing your environment. It covers dependencies, setup steps, and any required downloads. Follow it to get a clean local run.

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Installation

Clone the repository:

```bash
git clone https://github.com/your-username/transformer-from-scratch.git
cd transformer-from-scratch
```

Create and activate a virtual environment:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download NLTK data (required for evaluation):

```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
```

## Usage

This section shows how to use the project end to end. You will learn how to configure experiments, train the model, run inference, and evaluate results. Use it as your main workflow guide.

### Configuration

Model and training configurations are defined in YAML files:

Model configuration (`config/model.yaml`):

```yaml
src_vocab_size: 16000
tgt_vocab_size: 12000
d_model: 512
num_layers: 6
num_heads: 8
d_ff: 2048
max_seq_len: 40
dropout: 0.1
activation: gelu
```

Training configuration (`config/train.yaml`):

```yaml
batch_size: 32
epochs: 30
learning_rate: 1.0
warmup_steps: 4000
optimizer: adamw
weight_decay: 0.0001
label_smoothing: 0.1
gradient_clip_norm: 1.0
```

### Vocabulary

The project uses WordLevel tokenizers built with Hugging Face Tokenizers.

Special tokens:

| Token | Description |
|-------|-------------|
| `[PAD]` | Padding token |
| `[UNK]` | Unknown token |
| `[BOS]` | Beginning of sequence |
| `[EOS]` | End of sequence |

Build tokenizers from training data:

```bash
python -m tokenizer.build_vocab
```

Output:

| File | Language | Vocab Size |
|------|----------|------------|
| `tokenizer/en_tokenizer.json` | English | 16,000 |
| `tokenizer/vi_tokenizer.json` | Vietnamese | 12,000 |

Note: Pre-built tokenizers are already included in the repository. Rebuild only if you want to change the vocabulary size or dataset.

### Training

Train the Transformer model:

```bash
python train.py --device cuda --batch-size 32 --epochs 30
```

Arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--device` | Training device (`cuda` or `cpu`) | `cuda` |
| `--batch-size` | Override batch size from config | Config value |
| `--epochs` | Override number of epochs | Config value |
| `--resume` | Path to checkpoint for resuming | None |

Resume training from checkpoint:

```bash
python train.py --resume weights/last.pt
```

### Inference

Translate a sentence using the trained model:

```bash
python inference.py \
  --sentence "Hello, how are you?" \
  --checkpoint weights/best.pt \
  --strategy greedy \
  --device cpu
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--sentence` | Input English sentence | `"hello"` |
| `--checkpoint` | Path to model checkpoint | `weights/best.pt` |
| `--strategy` | Decoding strategy (`greedy` or `beam`) | `greedy` |
| `--beam_size` | Beam size for beam search | `5` |
| `--max_seq_len` | Maximum output sequence length | `40` |
| `--device` | Inference device | `cpu` |

### Evaluation

Evaluate the model on the test set:

```bash
python evaluate.py \
  --checkpoint weights/best.pt \
  --src-file data/iwslt2015_en_vi/test.en \
  --ref-file data/iwslt2015_en_vi/test.vi \
  --strategy greedy \
  --device cuda
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoint` | Path to trained model | Required |
| `--src-file` | Source sentences file | `data/iwslt2015_en_vi/test.en` |
| `--ref-file` | Reference translations file | `data/iwslt2015_en_vi/test.vi` |
| `--strategy` | Decoding strategy | `greedy` |
| `--beam-size` | Beam size for beam search | `4` |
| `--bert-model` | BERTScore model | `xlm-roberta-base` |
| `--device` | Evaluation device | `cpu` |

Evaluation metrics:

- BLEU
- chrF++
- BERTScore

## Experiments

This section is for documenting your training runs. Add plots, notes, and comparisons between experiments here. It makes your results easy to review and share.

Use this section to showcase training progress and compare experiments.

<p align="center">
  <img src="public/loss_plot.png" alt="Training Loss Curve" width="70%">
</p>

If you generate multiple plots, you can swap the file name or add additional images.

## License

This project is licensed under the MIT License. See `LICENSE`.

## References

This section lists the key papers and datasets behind this implementation. It gives proper credit and provides useful background reading. Check it if you want deeper context.

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [IWSLT/mt_eng_vietnamese](https://huggingface.co/datasets/IWSLT/mt_eng_vietnamese)
