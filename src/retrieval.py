import numpy as np

def exact_cosine_search(
    doc_ids,
    document_embeddings,
    query_embedding,
    k=10,
    exclude_ids=None,
):

    doc_ids = list(doc_ids)

    documents = np.asarray(
        document_embeddings,
        dtype=np.float32,
    )

    query = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    k = int(k)

    if k <= 0:
        return []

    scores = (documents @ query).copy()

    excluded = set(exclude_ids or [])

    excluded_indices = [
        i
        for i, doc_id in enumerate(doc_ids)
        if doc_id in excluded
    ]

    # masking before select top-k docs
    if excluded_indices:
        scores[excluded_indices] = -np.inf

    available = len(doc_ids) - len(excluded_indices)
    k_effective = min(k, available) 

    if k_effective <= 0:
        return []

    ranked_indices = np.argsort(
        -scores,
        kind="stable",
    )[:k_effective]

    return [
        doc_ids[i]
        for i in ranked_indices
    ]

