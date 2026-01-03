
```
transformer-mt/
│
├── config/
│   ├── train.yaml           # hyperparameters: batch_size, lr, warmup_steps, optimizer...
│   ├── model.yaml           # Transformer config: num_layers, d_model, n_heads, d_ff...
│
├── data/
│   ├── raw/                 # dataset gốc (downloaded)
│   │   ├── train.en
│   │   ├── train.vi
│   │   ├── valid.en
│   │   ├── valid.vi
│   │   └── test.en / test.vi
│   │
│   ├── processed/           # tokenized / preprocessed files
│   │   ├── train.pt
│   │   ├── valid.pt
│   │   └── test.pt
│   │
│   └── prepare_data.py      # download + preprocess + tokenize + save processed
│
├── tokenizer/
│   ├── vocab_builder.py     # BPE / SentencePiece / WordPiece training
│   └── tokenizer.py         # encode / decode + add special tokens
│
├── model/
│   ├── __init__.py
│   ├── layers.py            # MultiHeadAttention, FFN, PositionalEncoding
│   ├── encoder.py           # Transformer Encoder
│   ├── decoder.py           # Transformer Decoder
│   └── transformer.py       # Full Seq2Seq Transformer
│
├── loss/
│   ├── __init__.py
│   ├── label_smoothing.py   # label smoothing loss
│   └── masked_ce.py         # CrossEntropy with padding mask
│
├── engine/
│   ├── dataset.py           # PyTorch Dataset + collate_fn
│   ├── trainer.py           # Training loop: epochs, gradient clipping, logging
│   ├── train_step.py        # 1 forward-backward-update + optimizer step + scheduler step
│   └── evaluator.py         # validation / BLEU / perplexity
│
├── utils/
│   ├── masks.py             # create src_mask, tgt_mask, lookahead mask
│   ├── metrics.py           # BLEU, SacreBLEU, accuracy
│   ├── beam_search.py       # beam search decoding
│   ├── checkpoint.py        # save / load model + optimizer + scheduler
│   └── logging.py           # logger setup, experiment logging
│
├── inference/
│   ├── translate.py         # greedy / beam search translation
│   └── cli.py               # command-line interface for translation
│
├── experiments/             # lưu checkpoint, logs, BLEU outputs
│   ├── iwslt_en_vi/
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   └── translations/
│
├── tests/
│   ├── test_attention.py
│   ├── test_tokenizer.py
│   └── test_translation.py
│
├── train.py                 # main entrypoint: load config, dataloader, trainer, start training
├── evaluate.py              # run evaluation on test/valid, compute BLEU
├── inference.py             # quick inference script
├── requirements.txt
└── README.md

```

| Tham số          | Ý nghĩa                               | Gợi ý ban đầu                      |
| ---------------- | ------------------------------------- | ---------------------------------- |
| `num_layers`     | Số lớp encoder / decoder              | 6 (như paper gốc)                  |
| `d_model`        | Dimensionality của embedding / hidden | 512                                |
| `num_heads`      | Số attention head                     | 8                                  |
| `d_ff`           | Dim của feed-forward network          | 2048                               |
| `dropout_rate`   | Dropout trong attention / FFN         | 0.1                                |
| `max_seq_len`    | Max token sequence                    | 128–512 (tùy dataset)              |
| `vocab_size`     | Kích thước vocab                      | Tùy tokenizer (BPE, SentencePiece) |
| `activation`     | Activation trong FFN                  | ReLU (hoặc GELU)                   |
| `layer_norm_eps` | epsilon của LayerNorm                 | 1e-6                               |

| Tham số              | Ý nghĩa                     | Gợi ý                          |
| -------------------- | --------------------------- | ------------------------------ |
| `batch_size`         | Số mẫu / batch              | 32–128                         |
| `epochs`             | Số epoch                    | 50–100 (tùy dataset)           |
| `learning_rate`      | LR tối đa                   | 1e-4 / warmup schedule         |
| `warmup_steps`       | Steps warmup LR             | 4000 (như paper gốc)           |
| `optimizer`          | Optimizer                   | Adam + β1=0.9, β2=0.98, ε=1e-9 |
| `label_smoothing`    | Smoothing cho cross-entropy | 0.1                            |
| `gradient_clip_norm` | Clip gradient               | 1.0–5.0                        |
| `weight_decay`       | Regularization              | 0–1e-4                         |
