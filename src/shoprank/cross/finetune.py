"""Sections 12-13: fine-tune a cross-encoder (MiniLM, DeBERTa) on ESCI train.

Usage:
    uv run python -m shoprank.cross.finetune                          # 12a: MSE on gain, then scores dev
    uv run python -m shoprank.cross.finetune --loss ce                # 12b: 4-class CE + expected gain
    uv run python -m shoprank.cross.finetune --model microsoft/deberta-v3-base \
        --fields title,brand,color --lr 2e-5 --warmup-steps 1000 --eval-every 2500   # 13: DeBERTa
    uv run python -m shoprank.cross.finetune --limit-steps 200 --name minilm_ft_smoke   # timing check
    uv run python -m shoprank.cross.finetune --teacher deberta_oof --teacher-weight 1.0 --name minilm_distil
        # distillation: MiniLM regresses DeBERTa's out-of-fold scores instead of (or mixed with) the gain

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
CLASSES = ("E", "S", "C", "I")  # 4-class head: output i is the logit of CLASSES[i]
CLASS_GAINS = (1.0, 0.1, 0.01, 0.0)  # same gains as nDCG, in CLASSES order
LOSSES = ("mse", "ce")


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
    loss: str = "mse"  # "mse": 1 output, MSE on the gain; "ce": 4 outputs, cross-entropy on the class


def class_ids(labels: pd.Series) -> np.ndarray:
    """ESCI letters -> class index in CLASSES order (E 0, S 1, C 2, I 3)."""
    ids = labels.map({c: i for i, c in enumerate(CLASSES)})
    if ids.isna().any():
        raise ValueError(f"unknown ESCI labels: {sorted(set(labels[ids.isna()]))}")
    return ids.to_numpy(dtype=np.int64)


def score_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """One ranking score per row. One output: the output itself. Four outputs: the expected gain,
    P(E) * 1 + P(S) * 0.1 + P(C) * 0.01 + P(I) * 0, with probabilities from a softmax."""
    logits = logits.float()
    if logits.shape[-1] == 1:
        return logits.reshape(-1)
    gains = torch.tensor(CLASS_GAINS, device=logits.device)
    return logits.softmax(dim=-1) @ gains


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, loss: str) -> torch.Tensor:
    """"mse": targets are gains (float). "ce": targets are class indices (int)."""
    if loss == "mse":
        return torch.nn.functional.mse_loss(logits.float().reshape(-1), targets.float())
    if loss == "ce":
        return torch.nn.functional.cross_entropy(logits.float(), targets.long())
    raise ValueError(f"loss must be one of {LOSSES}, got {loss!r}")


def mix_targets(gain: np.ndarray, teacher: np.ndarray, weight: float) -> np.ndarray:
    """Distillation target: (1 - weight) * human gain + weight * teacher score. weight 1 = teacher only."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"teacher weight must be in [0, 1], got {weight}")
    gain, teacher = np.asarray(gain, dtype=np.float32), np.asarray(teacher, dtype=np.float32)
    if gain.shape != teacher.shape or not np.isfinite(teacher).all():
        raise ValueError("need one finite teacher score per training pair")
    return (1.0 - weight) * gain + weight * teacher


def warmup_linear(step: int, warmup: int, total: int) -> float:
    """LR multiplier: rises 0 -> 1 over `warmup` steps, then falls linearly to 0 at `total`."""
    if step < warmup:
        return step / max(1, warmup)
    return max(0.0, (total - step) / max(1, total - warmup))


def length_bucketed_batches(lengths: Sequence[int] | np.ndarray, batch_size: int, seed: int,
                            chunk_batches: int = 100) -> list[np.ndarray]:
    """One shuffled pass where each batch holds pairs of similar length, so little padding.

    Shuffle all rows, cut them into chunks of `chunk_batches` batches, sort each chunk by length, cut it
    into batches, then shuffle the batch order. Sorting only inside a chunk keeps batches random; the
    final shuffle stops the model seeing short pairs first and long pairs last.
    """
    lens = np.asarray(lengths)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(lens))
    chunk = batch_size * chunk_batches
    batches = []
    for start in range(0, len(order), chunk):
        part = order[start:start + chunk]
        part = part[np.argsort(lens[part], kind="stable")]
        batches.extend(part[i:i + batch_size] for i in range(0, len(part), batch_size))
    return [batches[i] for i in rng.permutation(len(batches))]


def pretokenize(tokenizer, queries: Sequence[str], texts: Sequence[str], max_length: int) -> dict:
    """Tokenise every (query, text) pair once, unpadded. Padding happens per batch in `collate`.
    Rows are stored as small int32 numpy arrays: 4 bytes per id instead of a Python int object + list
    pointer (about 36 bytes), which matters with 323k pairs on a 16 GB machine."""
    enc = tokenizer(list(queries), list(texts), truncation=True, max_length=max_length)
    return {k: [np.asarray(row, dtype=np.int32) for row in v] for k, v in enc.items()}


def model_tag(model_name: str) -> str:
    """Short run-name prefix: 'minilm' for the default model (keeps old run names), else the repo name."""
    if model_name == DEFAULT_MODEL:
        return "minilm"
    return model_name.split("/")[-1].lower().replace("-", "_")


def collate(features: dict, idx: np.ndarray, pad_id: int, device: str) -> dict:
    """Pad the rows in `idx` to the longest of them (not to max_length), as tensors on `device`."""
    width = max(len(features["input_ids"][i]) for i in idx)
    out = {}
    for key, rows in features.items():
        arr = np.full((len(idx), width), pad_id if key == "input_ids" else 0, dtype=np.int64)
        for r, i in enumerate(idx):
            arr[r, :len(rows[i])] = rows[i]
        out[key] = to_device(torch.from_numpy(arr), device)
    return out


def to_device(t: torch.Tensor, device: str) -> torch.Tensor:
    """Copy to the GPU without making the CPU wait. A copy from normal (pageable) RAM waits for all
    queued GPU work to finish; a copy from pinned RAM with non_blocking=True is queued like a kernel."""
    if not device.startswith("cuda"):
        return t.to(device)
    return t.pin_memory().to(device, non_blocking=True)


class Prepared4DMask(torch.nn.Module):
    """Gives a Hugging Face encoder a ready 4D boolean mask (batch, 1, 1, length), True = real token.

    With a 2D mask, transformers 5.23 runs `padding_mask.all()` inside a Python `if` to decide whether it
    can skip the mask (masking_utils.py:366). Reading that answer makes the CPU wait for the GPU every step.
    A 4D mask is used as-is (masking_utils.py:833), so there is no check and no wait. Same attention maths.
    Only for SDPA attention: "eager" attention ADDS the mask to the scores (modeling_bert.py:128), so a
    boolean mask would add 1 to real tokens and 0 to padding, i.e. no masking at all, silently.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        attn = getattr(getattr(model, "config", None), "_attn_implementation", "sdpa")
        if attn != "sdpa":
            raise ValueError(f"Prepared4DMask needs sdpa attention, got {attn!r}")
        self.model = model

    def forward(self, attention_mask: torch.Tensor, **inputs):
        return self.model(attention_mask=attention_mask[:, None, None, :].bool(), **inputs)


def _pad_id(tokenizer) -> int:
    return getattr(tokenizer, "pad_token_id", None) or 0


def _autocast(device: str, bf16: bool):
    return torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                          dtype=torch.bfloat16, enabled=bf16)


@torch.no_grad()
def predict(model, tokenizer, queries: Sequence[str], texts: Sequence[str], batch_size: int,
            max_length: int, device: str, bf16: bool = False) -> np.ndarray:
    """One relevance score per (query, text) pair, in input order. Pairs are scored in length order
    (less padding, same scores) and written back to their input positions."""
    if len(texts) == 0:
        return np.zeros(0)
    was_training = model.training
    model.eval()
    feats = pretokenize(tokenizer, queries, texts, max_length)
    order = np.argsort([len(x) for x in feats["input_ids"]], kind="stable")
    scores = np.zeros(len(order), dtype=np.float32)
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        with _autocast(device, bf16):
            logits = model(**collate(feats, idx, _pad_id(tokenizer), device)).logits
        scores[idx] = score_from_logits(logits).cpu().numpy()
    model.train(was_training)
    return scores


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
    """Train with `cfg.loss` (targets: gains for "mse", class indices for "ce"). Every `eval_every`
    steps, score the hold-out with `evaluate` (higher is better). At the end the model holds the
    best-scoring weights."""
    half = sorted({str(p.dtype) for p in model.parameters() if p.requires_grad and p.dtype != torch.float32})
    if half:
        # AdamW's eps (1e-8) rounds to 0 in fp16 and tiny squared gradients underflow, so the update is
        # 0/0 = NaN even at lr 0 (found with DeBERTa-v3, whose checkpoint is stored in fp16). Keep fp32
        # weights and let bf16 autocast speed up the maths.
        raise ValueError(f"trainable weights must be float32 for AdamW, found {half}; "
                         "load with from_pretrained(..., dtype=torch.float32)")
    torch.manual_seed(cfg.seed)
    feats = pretokenize(tokenizer, queries, texts, cfg.max_length)
    lengths = [len(x) for x in feats["input_ids"]]
    pad_id = _pad_id(tokenizer)
    epochs = [length_bucketed_batches(lengths, cfg.batch_size, cfg.seed + e) for e in range(cfg.epochs)]
    total = sum(len(b) for b in epochs)
    if cfg.limit_steps is not None:
        total = min(total, cfg.limit_steps)
    opt = _optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_linear(s, cfg.warmup_steps, total))
    y_all = torch.tensor(np.asarray(targets))  # a copy: pandas may hand over a read-only array

    history = {"evals": [], "losses": [], "best_step": 0, "train_seconds": 0.0}
    best_score, best_state = -float("inf"), None
    # The running loss stays on the device: calling .item() every step would make the CPU wait for the
    # GPU each step, so it could not prepare the next batch while the GPU works.
    step, run_loss, start = 0, torch.zeros((), device=device), time.perf_counter()
    model.train()
    for batches in epochs:
        for idx in batches:
            if step >= total:
                break
            enc = collate(feats, idx, pad_id, device)
            with _autocast(device, cfg.bf16):
                logits = model(**enc).logits
            loss = compute_loss(logits, to_device(y_all[idx], device), cfg.loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            sched.step()
            step += 1
            run_loss += loss.detach()

            if step % cfg.eval_every == 0 or step == total:
                mean_loss = run_loss.item() / cfg.eval_every  # .item() waits for the GPU to finish
                train_seconds = time.perf_counter() - start
                score = evaluate(model)
                history["evals"].append({"step": step, "score": score, "loss": mean_loss})
                log(f"step {step}/{total}  loss {mean_loss:.4f}  holdout nDCG {score:.4f}  "
                    f"lr {sched.get_last_lr()[0]:.2e}  {train_seconds:.0f}s")
                run_loss.zero_()
                if score > best_score:
                    best_score, history["best_step"] = score, step
                    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                start = time.perf_counter() - train_seconds  # eval time is not training time
    history["train_seconds"] = round(time.perf_counter() - start, 1)
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune a cross-encoder on ESCI train")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    p.add_argument("--fields", default="title", help="product fields joined by commas, e.g. title,brand,color")
    p.add_argument("--loss", choices=LOSSES, default="mse")
    p.add_argument("--lr", type=float, default=FitConfig.lr)
    p.add_argument("--warmup-steps", type=int, default=FitConfig.warmup_steps)
    p.add_argument("--seed", type=int, default=FitConfig.seed)
    p.add_argument("--name", default=None, help="default: <model tag>_ft_<loss>")
    p.add_argument("--limit-steps", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--teacher", default=None, help="OOF score set to distil from, e.g. deberta_oof (MSE only)")
    p.add_argument("--teacher-weight", type=float, default=1.0)
    args = p.parse_args()
    if args.teacher and args.loss != "mse":
        p.error("--teacher needs --loss mse")
    fields = args.fields.split(",")
    name = args.name or f"{model_tag(args.model)}_ft_{args.loss}"

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = FitConfig(lr=args.lr, warmup_steps=args.warmup_steps, eval_every=args.eval_every, seed=args.seed,
                    limit_steps=args.limit_steps, loss=args.loss)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Only the needed columns: descriptions and bullets are large and unused (saves RAM).
    products = pd.read_parquet(PROCESSED / "products.parquet", columns=["product_id", *fields])
    cols = ["query_id", "query", "product_id", "esci_label", "gain"]
    train = cached("train")[cols]
    hold = is_holdout(train["query_id"])
    fit_part, hold_part = train[~hold].reset_index(drop=True), train[hold].reset_index(drop=True)
    fit_texts = product_texts(fit_part, products, fields)
    hold_texts = product_texts(hold_part, products, fields)
    print(f"{args.model} | fields {fields} | fit {fit_part['query_id'].nunique():,} q / {len(fit_part):,} pairs; "
          f"hold-out {hold_part['query_id'].nunique():,} q / {len(hold_part):,} pairs; device {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # "ce" needs 4 outputs; the MS MARCO head has 1, so it is replaced by a new, randomly initialised head.
    # A model without a ranking head (DeBERTa) always gets a new random head.
    num_labels = 1 if cfg.loss == "mse" else len(CLASSES)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels, ignore_mismatched_sizes=num_labels != 1,
        dtype=torch.float32).to(device)  # fp32 master weights; DeBERTa-v3 is stored in fp16
    targets = fit_part["gain"].to_numpy() if cfg.loss == "mse" else class_ids(fit_part["esci_label"])
    if args.teacher:  # out-of-fold teacher scores: the teacher never trained on the query it scores
        from shoprank.ltr.stack import (
            load_oof_scores,  # lazy: ltr.stack imports this package
        )
        teacher = fit_part.merge(load_oof_scores(args.teacher, "train"), on=["query_id", "product_id"],
                                 how="left", validate="one_to_one")[args.teacher].to_numpy()
        targets = mix_targets(targets, teacher, args.teacher_weight)
    # The 4D-mask trick only applies to sdpa attention (BERT/MiniLM). DeBERTa has its own attention code,
    # which has no hidden GPU wait, so it runs unwrapped.
    sdpa = getattr(model.config, "_attn_implementation", None) == "sdpa"
    net = Prepared4DMask(model) if sdpa else model
    print(f"attention: {model.config._attn_implementation} | 4D-mask wrapper: {sdpa}", flush=True)

    def holdout_ndcg(m) -> float:
        s = predict(m, tokenizer, hold_part["query"].tolist(), hold_texts, 128, cfg.max_length, device, cfg.bf16)
        return float(np.mean([r["ndcg_full"] for r in per_query_results(hold_part.assign(score=s), "score")]))

    history = fit(net, tokenizer, fit_part["query"].tolist(), fit_texts, targets,
                  cfg, device, holdout_ndcg, log=lambda s: print(s, flush=True))
    out_dir = MODELS / name
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    dev = cached("dev")[cols]
    start = time.perf_counter()
    scores = predict(net, tokenizer, dev["query"].tolist(), product_texts(dev, products, fields),
                     128, cfg.max_length, device, cfg.bf16)
    config = {**asdict(cfg), "model": args.model, "fields": fields,
              "loss": "mse_gain" if cfg.loss == "mse" else "ce_4class_expected_gain",
              "best_step": history["best_step"], "train_seconds": history["train_seconds"],
              "teacher": args.teacher, "teacher_weight": args.teacher_weight if args.teacher else None,
              "fit_pairs": len(fit_part), "dev_score_seconds": round(time.perf_counter() - start, 1),
              "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else 0.0}
    summary = save(name, "dev", per_query_results(dev.assign(score=scores), "score"), config)
    print(f"{name}: dev nDCG full {summary['ndcg_full']:.4f}  nDCG@10 {summary['ndcg10']:.4f}  "
          f"best step {history['best_step']}  train {history['train_seconds']:.0f}s  "
          f"peak VRAM {config['peak_vram_gb']} GB", flush=True)


if __name__ == "__main__":
    main()
