"""Sanity tests for the public-release version of the experiment scripts.

These tests are intentionally lightweight: they verify that the
configurable-paths refactor and the silent-Telegram refactor are correct,
without trying to actually train a model (which would require TensorFlow,
PyTorch, an 8-GPU server and a 2.4-million-image dataset).

The tests run on any machine with stdlib Python — no heavy ML deps needed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ACTIVE_SCRIPTS = [
    REPO_ROOT / "CNN_Test_10.py",
    REPO_ROOT / "CNN_Test_11.py",
    REPO_ROOT / "CNN_Test_12.py",
    REPO_ROOT / "ViT_Test_7.py",
]
TEMP_HELPERS = [
    REPO_ROOT / "~temp" / "check_gpu.py",
    REPO_ROOT / "~temp" / "psnr_lsb.py",
    REPO_ROOT / "~temp" / "psnr_dct.py",
    REPO_ROOT / "~temp" / "psnr_dwt.py",
    REPO_ROOT / "~temp" / "plot_csv.py",
]


@pytest.mark.parametrize("script", ACTIVE_SCRIPTS + TEMP_HELPERS, ids=lambda p: p.name)
def test_script_parses(script: Path) -> None:
    """Every shipped Python script must be syntactically valid."""
    assert script.exists(), f"missing file: {script}"
    ast.parse(script.read_text())


@pytest.mark.parametrize("script", ACTIVE_SCRIPTS, ids=lambda p: p.name)
def test_no_hardcoded_absolute_paths(script: Path) -> None:
    """Active scripts should not contain hardcoded /home/srojas or /HDDmedia
    paths outside of the env-var defaults block."""
    src = script.read_text()
    # The only allowed occurrences are inside the os.environ.get(...) defaults,
    # which look like:  TG2_HOME = os.environ.get("TG2_HOME", "/home/srojas/tg2")
    # ... but we replaced those with "." defaults, so neither pattern should appear.
    assert "/home/srojas" not in src, f"hardcoded /home/srojas in {script.name}"
    assert "/HDDmedia/srojas" not in src, f"hardcoded /HDDmedia in {script.name}"


@pytest.mark.parametrize("script", ACTIVE_SCRIPTS, ids=lambda p: p.name)
def test_default_tg2_home_is_relative(script: Path) -> None:
    """Default TG2_HOME should be ``.`` so the scripts work out-of-the-box."""
    src = script.read_text()
    assert 'os.environ.get("TG2_HOME", ".")' in src, (
        f"{script.name} default TG2_HOME is not '.'"
    )


@pytest.mark.parametrize("script", ACTIVE_SCRIPTS, ids=lambda p: p.name)
def test_default_tg2_data_root_is_relative(script: Path) -> None:
    """Default TG2_DATA_ROOT should be ``./data``."""
    src = script.read_text()
    assert 'os.environ.get("TG2_DATA_ROOT", "./data")' in src, (
        f"{script.name} default TG2_DATA_ROOT is not './data'"
    )


@pytest.mark.parametrize("script", ACTIVE_SCRIPTS, ids=lambda p: p.name)
def test_telegram_function_is_silent_when_unconfigured(script: Path) -> None:
    """The refactored send_telegram_message must early-return when CONFIG_INI
    is missing — guard against regressions to the old print-warning behavior."""
    src = script.read_text()
    assert "if not os.path.exists(CONFIG_INI):" in src, (
        f"{script.name} send_telegram_message lacks the silent-no-op guard"
    )


def test_no_telegram_token_pattern_in_source() -> None:
    """No bot-token-shaped literal should appear in the published scripts.

    Telegram bot tokens have the format ``<numeric_id>:<35-char-base64ish>``.
    This regex catches anything that looks like one. The test deliberately
    does NOT hardcode any specific token so that the test source itself is
    safe to publish.
    """
    import re

    token_pattern = re.compile(r"\b\d{8,}:[A-Za-z0-9_-]{30,}\b")
    for script in ACTIVE_SCRIPTS + TEMP_HELPERS:
        if not script.exists():
            continue
        match = token_pattern.search(script.read_text())
        assert match is None, (
            f"{script.name} contains a string that looks like a Telegram bot token"
        )
