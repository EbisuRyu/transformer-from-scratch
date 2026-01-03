
```
transformer-mt/
│
├── config/
│   ├── train_config.yaml         # hyperparams, optimizer, scheduler 
│   ├── model_config.yaml         # num_layers, d_model, n_heads,...
│
├── data/
│   ├── raw/                      # dataset gốc
│   ├── processed/                # sau preprocess/tokenize
│   ├── prepare_data.py           # download + preprocess + build vocab
│
├── tokenizer/
│   ├── vocab_builder.py          # BPE/WordPiece/Sentencepiece training
│   ├── tokenizer.py              # encode/decode, add special tokens
│
├── model/
│   ├── layers.py                 # MultiHead Attention, FFN, Positional Encoding
│   ├── encoder.py                # Transformer Encoder
│   ├── decoder.py                # Transformer Decoder
│   ├── transformer.py            # Full seq2seq Transformer
│
├── loss/
│   ├── label_smoothing.py
│   ├── masked_loss.py            # cross entropy mask padding
│
├── utils/
│   ├── masks.py                  # padding mask, lookahead mask
│   ├── metrics.py                # BLEU, SacreBLEU
│   ├── beam_search.py
│   ├── logging.py
│   ├── checkpoint.py
│
├── training/
│   ├── dataset.py                # PyTorch dataset + collate_fn
│   ├── dataloader.py
│   ├── train_step.py             # 1 forward-backward-update
│   ├── trainer.py                # training loop full
│
├── inference/
│   ├── translate.py              # greedy / beam search decoding
│   ├── cli.py                    # command-line translate interface
│
├── notebooks/
│   ├── model_sanity_check.ipynb
│   ├── inference_demo.ipynb
│
├── tests/
│   ├── test_attention.py
│   ├── test_tokenizer.py
│   ├── test_translation.py
│
├── train.py                      # main training entrypoint
├── evaluate.py                   # compute BLEU, generate outputs
├── inference.py                  # quick inference script
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
