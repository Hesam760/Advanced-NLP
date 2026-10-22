import math

def retrieval_metrics(ranked, gold_ids):
    gold = set(gold_ids)

    if not gold:
        raise ValueError("gold_ids cannot be empty")

    result = {}

    for k in (2, 5, 10):
        top_k = set(ranked[:k])

        result[f"recall@{k}"] = (
            len(top_k & gold) / len(gold)
        )

        result[f"hitall@{k}"] = float(
            gold.issubset(top_k)
        )

    result["mrr"] = next(
        (
            1.0 / rank
            for rank, doc_id in enumerate(ranked, start=1)
            if doc_id in gold
        ),
        0.0,
    )

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:10], start=1)
        if doc_id in gold
    )

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(gold), 10) + 1)
    )

    result["ndcg@10"] = dcg / idcg

    return result