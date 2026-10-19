
import math
import pytest

from src.metrics import retrieval_metrics
from score import per_question


def test_perfect_retrieval():
    result = retrieval_metrics(
        ranked=["g1", "g2", "x"],
        gold_ids=["g1", "g2"],
    )

    assert result["recall@2"] == 1.0
    assert result["hitall@2"] == 1.0
    assert result["mrr"] == 1.0
    assert result["ndcg@10"] == pytest.approx(1.0)


def test_partial_retrieval():
    result = retrieval_metrics(
        ranked=["x", "g1", "y", "g2"],
        gold_ids=["g1", "g2"],
    )

    assert result["recall@2"] == 0.5
    assert result["hitall@2"] == 0.0
    assert result["recall@5"] == 1.0
    assert result["hitall@5"] == 1.0
    assert result["mrr"] == 0.5

    expected_dcg = (
        1 / math.log2(3)
        + 1 / math.log2(5)
    )
    expected_idcg = 1 + 1 / math.log2(3)

    assert result["ndcg@10"] == pytest.approx(
        expected_dcg / expected_idcg
    )


def test_no_relevant_documents():
    result = retrieval_metrics(
        ranked=["x", "y", "z"],
        gold_ids=["g1"],
    )

    assert all(value == 0.0 for value in result.values())


def test_empty_gold_is_rejected():
    with pytest.raises(ValueError):
        retrieval_metrics(["x"], [])


def test_agreement_with_official_scorer():
    prediction = {
        "qid": "q1",
        "answer": "",
        "retrieved": ["x", "g1", "y", "g2"],
    }

    label = {
        "qid": "q1",
        "answer": "example",
        "gold_doc_ids": ["g1", "g2"],
    }

    ours = retrieval_metrics(
        prediction["retrieved"],
        label["gold_doc_ids"],
    )

    official = per_question(prediction, label)

    for metric_name, value in ours.items():
        assert value == pytest.approx(
            official[metric_name]
        )
