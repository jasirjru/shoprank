"""Pinned Hugging Face revisions: known ids are pinned, local folders need none, unknown ids warn."""
from __future__ import annotations

import logging
import re

from shoprank.cross.zero_shot import DEFAULT_MODEL
from shoprank.hub import HUB_REVISIONS, revision_for


def test_every_pin_is_a_full_commit_hash():
    assert HUB_REVISIONS
    assert all(re.fullmatch(r"[0-9a-f]{40}", rev) for rev in HUB_REVISIONS.values())


def test_default_and_teacher_models_are_pinned():
    assert revision_for(DEFAULT_MODEL) == HUB_REVISIONS[DEFAULT_MODEL]
    assert revision_for("microsoft/deberta-v3-base") is not None


def test_local_folder_needs_no_revision(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="shoprank.hub"):
        assert revision_for(str(tmp_path)) is None
    assert not caplog.records


def test_unpinned_hub_id_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="shoprank.hub"):
        assert revision_for("someone/unknown-model") is None
    assert "no pinned revision" in caplog.text
