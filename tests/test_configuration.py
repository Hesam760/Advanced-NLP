from pathlib import Path
from src.config import load_config
ROOT = Path(__file__).resolve().parents[1]

def test_config_files_and_prompts():
    for name in ("closed_book", "oracle", "dense"):
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        assert cfg["phase"] == 1
        prompt = ROOT / cfg["settings"]["prompt_file"]
        assert prompt.is_file()
        assert prompt.read_text(encoding="utf-8").strip()
        assert cfg["settings"]["generation"]["max_new_tokens"] > 0


def test_models_and_dense_retrieval_config():
    cfgs = [load_config(ROOT / "configs" / f"{n}.yaml") for n in ("closed_book", "oracle", "dense")]
    assert len({c["models"]["llm"]["name"] for c in cfgs}) == 1
    dense = cfgs[-1]
    assert dense["settings"]["retrieval"]["document_mask"] is True
    assert dense["settings"]["retrieval"]["k"] >= 10
    assert dense["settings"]["retrieval"]["method"] == "exact_cosine"


def test_notebook_imports_tested_modules():
    import json
    path = ROOT / "advanced-nlp.ipynb"
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    code = "\n".join("".join(c.get("source", [])) for c in cells)
    assert "from src.metrics import retrieval_metrics" in code
    assert "from src.retrieval import exact_cosine_search" in code
    assert "load_prompt(DENSE_CFG)" in code
