"""Shared pytest fixtures for hume2md tests.

Author: Alister Lewis-Bowen <alister@lewis-bowen.org>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hume2md import Token  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def load_tokens(name: str) -> list[Token]:
    """Load an OCR-token fixture by filename stem.

    Args:
        name: Fixture filename without the ``.json`` extension.

    Returns:
        The fixture's tokens, in file order.
    """
    data = json.loads((FIXTURES_DIR / f"{name}.json").read_text())
    return [Token(text=d["text"], x=d["x"], y=d["y"]) for d in data]
