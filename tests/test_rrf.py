import pytest
from src.retrieval import reciprocal_rank_fusion


def test_rrf_fuses_two_ranked_lists():
    ranked_lists = [
        ["A", "B", "C"],
        ["B", "D", "A"],
    ]

    result = reciprocal_rank_fusion(ranked_lists)

    assert result == ["B", "A", "D", "C"]


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_duplicate_document_is_counted_once_per_list():
    result = reciprocal_rank_fusion(
        [["A", "A", "B"], ["B", "C"]]
    )

    assert len(result) == len(set(result))
    assert set(result) == {"A", "B", "C"}


def test_rrf_rejects_negative_rank_constant():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            [["A", "B"]],
            rank_constant=-1,
        )