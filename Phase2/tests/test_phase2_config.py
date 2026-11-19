from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_phase2_config_is_structured_and_matches_saved_hybrid_run():
    cfg = yaml.safe_load((ROOT / "configs/phase2.yaml").read_text(encoding="utf-8"))
    assert cfg["phase"] == 2
    assert cfg["generation"]["do_sample"] is False
    assert cfg["generation"]["temperature"] == 0
    assert cfg["retrieval"]["bm25_k1"] == 1.5
    assert cfg["retrieval"]["bm25_b"] == 0.75
    assert cfg["retrieval"]["rrf_k"] == 60
    assert cfg["runs"]["dev_hybrid"]["method"] == "hybrid_rrf"
    assert cfg["runs"]["dev_masked"]["gold_masked_before_retrieval"] is True
    assert cfg["runs"]["dev_dense_fair"]["method"] == "dense_exact_cosine"
    assert cfg["runs"]["dev_hybrid"]["dense_top_k"] == cfg["runs"]["hidden_hybrid"]["dense_top_k"]
