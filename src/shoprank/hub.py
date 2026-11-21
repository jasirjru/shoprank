"""Pinned Hugging Face Hub revisions for every base model, so an upstream push cannot silently change results.

The commits are the ones the v1.0 runs downloaded (Hugging Face cache, refs/main, 18 Nov 2026).
DeBERTa-v3-base keeps its weights as pytorch_model.bin on main; transformers loads the Hub's automatic
safetensors conversion of the same weights (cache snapshot de19fe7), so config, tokenizer and weights all
trace back to this main commit.
A local directory (a fine-tuned model under data/processed/models/) needs no revision.
"""
from __future__ import annotations

import logging
from pathlib import Path

LOG = logging.getLogger("shoprank.hub")

HUB_REVISIONS: dict[str, str] = {
    "cross-encoder/ms-marco-MiniLM-L-12-v2": "7b0235231ca2674cb8ca8f022859a6eba2b1c968",
    "microsoft/deberta-v3-base": "8ccc9b6f36199bec6961081d44eb72fb3f7353f3",
}


def revision_for(model: str) -> str | None:
    """Pinned commit for a Hub model id; None for a local directory or an unpinned id (with a warning)."""
    if model in HUB_REVISIONS:
        return HUB_REVISIONS[model]
    if not Path(model).exists():
        LOG.warning("model %s has no pinned revision: results may change if the Hub copy changes", model)
    return None
