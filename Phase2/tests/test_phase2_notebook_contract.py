import json
import re
from pathlib import Path

from src import phase2_core


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Advanced_NLP_Phase2.ipynb"


def notebook_code():
    with NOTEBOOK.open(encoding="utf-8") as f:
        notebook = json.load(f)

    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_notebook_exists_and_is_valid():
    with NOTEBOOK.open(encoding="utf-8") as f:
        notebook = json.load(f)

    assert notebook["nbformat"] == 4
    assert any(
        cell["cell_type"] == "code"
        for cell in notebook["cells"]
    )


def test_notebook_imports_shared_core():
    code = notebook_code()

    assert "src.phase2_core" in code


def test_shared_core_implementations_exist():
    required = [
        "BM25",
        "bm25_tokens",
        "rrf_fusion",
        "parse_grounded_output",
        "passage_block",
        "pack_context",
        "cosine_rank",
    ]

    for name in required:
        assert callable(getattr(phase2_core, name))


def test_notebook_has_no_duplicate_core_implementations():
    code = notebook_code()

    duplicated = [
        r"^\s*class\s+BM25\b",
        r"^\s*def\s+bm25_tokens\b",
        r"^\s*def\s+rrf_fusion\b",
        r"^\s*def\s+parse_grounded_output\b",
    ]

    for pattern in duplicated:
        assert re.search(pattern, code, re.MULTILINE) is None
