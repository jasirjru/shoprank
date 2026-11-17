# ShopRank stack (decided 18 Nov 2026)

Rule: a tool gets in only if it moves the metric, makes results reproducible, or is needed for the next section. Everything is project-local (`uv add`, pinned in `uv.lock`). Versions were checked on PyPI on 17-18 Nov 2026.

## Machine (measured)

- GPU: NVIDIA GeForce RTX 4060 Laptop, 8 GB (driver 610.62), CUDA 12.8
- CPU: AMD Ryzen 9 7940HX; RAM: 15 GB usable
- Disk: C: 77 GB free (92% used); keep model checkpoints pruned

## In use

| Layer | Tool | Version | Why |
|---|---|---|---|
| Environment | uv | – | Exact versions on every machine (`uv.lock`) |
| Data | DuckDB + Parquet (pyarrow) | 2.0.0 / 26.0.1 | SQL on 1.1 GB of files, no server, fits in RAM |
| Numerics | NumPy, pandas | 2.5.4 / 3.1.1 | Bootstrap, feature tables |
| Classic ML | scikit-learn | 1.9.1 | Reference logistic regression |
| Learning to rank | LightGBM (LambdaMART) | 4.7.0 | Best model for tabular ranking features; later the stacker |
| Deep learning | PyTorch (CUDA 12.8 build) | 2.11.0+cu128 | GPU training |
| Transformers | transformers | 5.23.0 | DeBERTa-v3 models |
| Cross-encoder training | sentence-transformers | 6.3.0 | `CrossEncoderTrainer`, ranking losses, `mine_hard_negatives()`, `CrossEncoderRerankingEvaluator` |
| Training plumbing | datasets, accelerate | 5.2.0 / 1.16.0 | Required by the trainer |
| Tokeniser | sentencepiece, protobuf | 0.2.3 / 7.36.2 | DeBERTa-v3 tokeniser |
| Official metric | pytrec-eval-terrier | 0.5.11 | Outside check of our nDCG (see below) |
| Tests and lint | pytest, Ruff | 9.2 / 0.17 | – |

## Measured on this GPU (18 Nov 2026)

Settings: bf16 autocast, sequence length 128, AdamW, 4 output classes; median step time over 10 steps.

| Model | Batch | Gradient checkpointing | Step | Throughput | Peak VRAM | Verdict |
|---|---|---|---|---|---|---|
| **DeBERTa-v3-base (184M)** | **32** | no | **244 ms** | **131 pairs/s** | **5.1 GB** | **Use this** |
| DeBERTa-v3-base | 64 | no | 2,757 ms | 23 pairs/s | 8.9 GB | Exceeds 8 GB and spills to shared memory, so it's 6x slower. Avoid. |
| DeBERTa-v3-large (435M) | 8 | yes | 237 ms | 34 pairs/s | 5.7 GB | Works, about 4x slower than base |
| DeBERTa-v3-large | 16 | yes | 1,709 ms | 9 pairs/s | 8.2 GB | Spills. Avoid. |

Estimated time per epoch on about 377k US Task 1 training pairs: **base about 48 min; large about 3.1 h**. These are estimates from the throughput above, not measured epochs.

## Metric check

On 500 random queries, `pytrec_eval` `ndcg` (gains E/S/C/I = 100/10/1/0, which is 100x the official gains; nDCG doesn't change under uniform scaling) equals ShopRank's `ndcg_at_k` with a maximum per-query difference of 5.55e-16. Our metric matches the official trec_eval metric.

Note: pytrec_eval can't parse the parameter string `ndcg.1=0,2=0.01,3=0.1,4=1`, because it splits on the dot. Pass integer gains as relevance levels instead.

## Add later (when its section starts)

| Tool | Version | When |
|---|---|---|
| gradio | 6.32.0 | Phase C live demo |
| optuna | 5.1.0 | Phase B4 hyperparameter search |
| onnxruntime-gpu | 1.32.0 | Only if the demo needs faster inference |
| faiss-cpu | 1.16.0 | Only for full-catalogue retrieval (not needed for Task 1 re-ranking) |

## Skipped, and why

| Tool | Why not |
|---|---|
| XGBoost, CatBoost | The same family as LightGBM; it adds nothing to measure |
| MLflow, Weights & Biases | The lab notebook plus run JSON files are enough for one person; it would add infrastructure |
| flash-attn | Hard to build on Windows, and DeBERTa's attention doesn't use it |
| bitsandbytes, PEFT (QLoRA) | Base and large fine-tune fully on 8 GB; only needed for the optional LLM query parser |
| polars, bm25s, ranx | DuckDB, our own BM25 and pytrec_eval already cover these |
