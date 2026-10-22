import numpy as np
import pytest
from src.retrieval import exact_cosine_search

DOC_IDS = ["p1", "p2", "p3"]
DOC_EMBEDDINGS = np.array([
    [1.0, 0.0],
    [0.8, 0.6],
    [0.6, 0.8],
], dtype=np.float32)

QUERY = np.array([1.0, 0.0], dtype=np.float32)

def search(k, exclude_ids=None):
    return exact_cosine_search(
        DOC_IDS,
        DOC_EMBEDDINGS,
        QUERY,
        k=k,
        exclude_ids=exclude_ids,
    )


def test_exact_cosine_ranking():
    assert search(k=3) == ["p1", "p2", "p3"]


def test_document_mask_before_topk():
    result = search(
        k=2,
        exclude_ids={"p1"},
    )
    assert result == ["p2", "p3"]


def test_multiple_exclusions():
    assert search(
        k=2,
        exclude_ids={"p1", "p2"},
    ) == ["p3"]


def test_all_documents_excluded():
    assert search(
        k=2,
        exclude_ids=set(DOC_IDS),
    ) == []


def test_k_larger_than_corpus():
    assert search(k=100) == DOC_IDS


def test_zero_k():
    assert search(k=0) == []


def test_unknown_exclusion_is_ignored():
    assert search(
        k=2,
        exclude_ids={"unknown"},
    ) == ["p1", "p2"]


def test_stable_tie_breaking():
    documents = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
    ])

    result = exact_cosine_search(
        ["first", "second"],
        documents,
        QUERY,
        k=2,
    )

    assert result == ["first", "second"]


def test_embedding_dimension_mismatch():
    with pytest.raises(ValueError):
        exact_cosine_search(
            ["p1"],
            [[1.0, 0.0]],
            [1.0, 0.0, 0.0],
        )
