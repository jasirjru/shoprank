"""Section 12a: fine-tune the MiniLM cross-encoder on ESCI train (Amazon's baseline recipe).

Usage:
    uv run python -m shoprank.cross.finetune                          # full run, then scores dev
    uv run python -m shoprank.cross.finetune --limit-steps 200 --name minilm_ft_smoke   # timing check

Recipe (amazon-science/esci-data ranking/train.py): ms-marco-MiniLM-L-12-v2, one output, no activation,
MSE on the gain (E 1, S 0.1, C 0.01, I 0), 1 epoch, lr 7e-6, 5,000 warmup steps, batch 32, title only,
seed 42. Ours: max_length 128 (<0.01% of train pairs are longer), bf16 autocast, and checkpoint choice on
the TRAIN hold-out (same 1-in-10 query hash as LambdaMART), never on dev.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

from shoprank.cross.zero_shot import DEFAULT_MODEL, product_texts
from shoprank.eval.run_task1 import save
from shoprank.features.build import PROCESSED, cached
from shoprank.ltr.train import is_holdout, per_query_results

MODELS = PROCESSED / "models"
NO_DECAY = ("bias", "LayerNorm.weight", "LayerNorm.bias")  # old CrossEncoder.fit default


@dataclass(frozen=True)
class FitConfig:
    lr: float = 7e-6
    warmup_steps: int = 5000
    batch_size: int = 32
    epochs: int = 1
    eval_every: int = 1000
    max_length: int = 128
    seed: int = 42
    bf16: bool = True
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    limit_steps: int | None = None  # stop early (timing / smoke runs only)


def warmup_linear(step: int, warmup: int, total: int) -> float:
    """LR multiplier: rises 0 -> 1 over `warmup` steps, then falls linearly to 0 at `total`."""
    if step < warmup:
        return step / max(1, warmup)
    return max(0.0, (total - step) / max(1, total - warmup))


def batch_indices(n: int, batch_size: int, seed: int) -> list[np.ndarray]:
    """One shuffled pass over n rows, cut into batches (the last one may be short)."""
    order = np.random.default_rng(seed).permutation(n)
    return [order[i:i + batch_size] for i in range(0, n, batch_size)]


def _encode(tokenizer, queries: Sequence[str], texts: Sequence[str], max_length: int, device: str) -> dict:
    enc = tokenizer(list(queries), list(texts), padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def _autocast(device: str, bf16: bool):
    return torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                          dtype=torch.bfloat16, enabled=bf16)


@torch.no_grad()
def predict(model, tokenizer, queries: Sequence[str], texts: Sequence[str], batch_size: int,
            max_length: int, device: str, bf16: bool = False) -> np.ndarray:
    """One relevance score per (query, text) pair, in input order."""
    was_training = model.training
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = _encode(tokenizer, queries[i:i + batch_size], texts[i:i + batch_size], max_length, device)
        with _autocast(device, bf16):
            out.append(model(**enc).logits.float().reshape(-1).cpu())
    model.train(was_training)
    return torch.cat(out).numpy() if out else np.zeros(0)


def _optimizer(model, cfg: FitConfig) -> torch.optim.AdamW:
    params = list(model.named_parameters())
    groups = [
        {"params": [p for n, p in params if not any(nd in n for nd in NO_DECAY)], "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in params if any(nd in n for nd in NO_DECAY)], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr)


def fit(model, tokenizer, queries: Sequence[str], texts: Sequence[str], targets: np.ndarray,
        cfg: FitConfig, device: str, evaluate: Callable[[torch.nn.Module], float],
        log: Callable[[str], None] = lambda s: None) -> dict:
    """Train with MSE on the gain. Every `eval_every` steps, score the hold-out with `evaluate`
    (higher is better). At the end the model holds the best-scoring weights."""
    torch.manual_seed(cfg.seed)
    epochs = [batch_indices(len(texts), cfg.batch_size, cfg.seed + e) for e in range(cfg.epochs)]
    total = sum(len(b) for b in epochs)
    if cfg.limit_steps is not None:
        total = min(total, cfg.limit_steps)
    opt = _optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_linear(s, cfg.warmup_steps, total))
    y_all = torch.tensor(np.asarray(targets), dtype=torch.float32)

    history = {"evals": [], "losses": [], "best_step": 0, "train_seconds": 0.0}
    best_score, best_state = -float("inf"), None
    step, run_loss, start = 0, 0.0, time.perf_counter()
    model.train()
    for batches in epochs:
        for idx in batches:
            if step >= total:
                break
            enc = _encode(tokenizer, [queries[i] for i in idx], [texts[i] for i in idx], cfg.max_length, device)
            with _autocast(device, cfg.bf16):
                logits = model(**enc).logits
            loss = torch.nn.functional.mse_loss(logits.float().reshape(-1), y_all[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            sched.step()
            step += 1
            run_loss += loss.item()

            if step % cfg.eval_every == 0 or step == total:
                train_seconds = time.perf_counter() - start
                score = evaluate(model)
                history["evals"].append({"step": step, "score": score, "loss": run_loss / cfg.eval_every})
                log(f"step {step}/{total}  loss {run_loss / cfg.eval_every:.4f}  holdout nDCG {score:.4f}  "
                    f"lr {sched.get_last_lr()[0]:.2e}  {train_seconds:.0f}s")
                run_loss = 0.0
                if score > best_score:
                    best_score, history["best_step"] = score, step
                    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                start = time.perf_counter() - train_seconds  # eval time is not training time
    history["train_seconds"] = round(time.perf_counter() - start, 1)
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune MiniLM cross-encoder on ESCI train (MSE on gain)")
    p.add_argument("--name", default="minilm_ft_mse")
    p.add_argument("--limit-steps", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=1000)
    args = p.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = FitConfig(eval_every=args.eval_every, limit_steps=args.limit_steps)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    products = pd.read_parquet(PROCESSED / "products.parquet")
    cols = ["query_id", "query", "product_id", "esci_label", "gain"]
    train = cached("train")[cols]
    hold = is_holdout(train["query_id"])
    fit_part, hold_part = train[~hold].reset_index(drop=True), train[hold].reset_index(drop=True)
    fit_texts = product_texts(fit_part, products, ["title"])
    hold_texts = product_texts(hold_part, products, ["title"])
    print(f"fit {fit_part['query_id'].nunique():,} q / {len(fit_part):,} pairs; "
          f"hold-out {hold_part['query_id'].nunique():,} q / {len(hold_part):,} pairs; device {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL, num_labels=1).to(device)

    def holdout_ndcg(m) -> float:
        s = predict(m, tokenizer, hold_part["query"].tolist(), hold_texts, 128, cfg.max_length, device, cfg.bf16)
        return float(np.mean([r["ndcg_full"] for r in per_query_results(hold_part.assign(score=s), "score")]))

    history = fit(model, tokenizer, fit_part["query"].tolist(), fit_texts, fit_part["gain"].to_numpy(),
                  cfg, device, holdout_ndcg, log=lambda s: print(s, flush=True))
    out_dir = MODELS / args.name
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    dev = cached("dev")[cols]
    start = time.perf_counter()
    scores = predict(model, tokenizer, dev["query"].tolist(), product_texts(dev, products, ["title"]),
                     128, cfg.max_length, device, cfg.bf16)
    config = {**asdict(cfg), "model": DEFAULT_MODEL, "fields": ["title"], "loss": "mse_gain",
              "best_step": history["best_step"], "train_seconds": history["train_seconds"],
              "fit_pairs": len(fit_part), "dev_score_seconds": round(time.perf_counter() - start, 1)}
    summary = save(args.name, "dev", per_query_results(dev.assign(score=scores), "score"), config)
    print(f"{args.name}: dev nDCG full {summary['ndcg_full']:.4f}  nDCG@10 {summary['ndcg10']:.4f}  "
          f"best step {history['best_step']}  train {history['train_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
