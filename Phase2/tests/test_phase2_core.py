import json
import math
from pathlib import Path

from src.phase2_core import BM25, bm25_tokens, make_units, parse_grounded_output, rrf_fusion


def test_positive_idf_and_oov():
    b = BM25([('d1', 'apple apple banana'), ('d2', 'banana pear')], k1=1.5,b=.75)
    assert b.idf['banana'] > 0
    assert math.isclose(b.idf['apple'], math.log1p((2-1+.5)/(1+.5)))
    assert b.search('apple')==['d1']
    assert b.search('neverseenword')==[]


def test_mask_applied_before_topk():
    b = BM25([('d1','apple apple apple'),('d2','apple'),('d3','pear')])
    assert b.search('apple',k=1)==['d1']
    assert b.search('apple',k=1,exclude_ids={'d1'})==['d2']
    assert b.search('apple',k=10,exclude_ids={'d1','d2'})==[]


def test_rrf_rank_starts_at_one_and_ties_deterministically():
    assert rrf_fusion([['b','a'],['a','b']],k=2)==['a','b']
    assert rrf_fusion([['a','a'],['b']],k=2)==['a','b']
    assert rrf_fusion([['c'],['c']],k=1)==['c']
    assert len(rrf_fusion([['a'],['a']],k=3))==1


def test_parser_citations_thinking_and_broken_json():
    allowed=['d1','d2']
    raw='<think>example</think> Answer: {"answer":"Paris","cited":["d1","d1"],"abstained":false}'
    assert parse_grounded_output(raw,allowed)=={'answer':'Paris','cited':['d1'],'abstained':False}
    assert parse_grounded_output('{"answer":"Paris","cited":["d3"],"abstained":false}',allowed)['abstained']
    assert parse_grounded_output('garbage',allowed)['abstained']
    assert parse_grounded_output('{"answer":"","cited":[],"abstained":true}',allowed)['abstained']


def test_sentence_units_keep_source_doc_id():
    c=[{'doc_id':'d1','title':'Thing','text':'Alpha. Beta.', 'sentences':['Alpha.','Beta.']}]
    units=make_units(c,'sentence')
    assert units==[('d1','Alpha.'),('d1','Beta.')]
    assert make_units(c,'title')[0][1].startswith('Thing\n')


def test_reference_bm25_when_available():
    import pytest
    reference = pytest.importorskip('rank_bm25')
    docs=[('a','apple apple banana'),('b','banana and pear'),('c','apple pear pear')]
    toks=[bm25_tokens(t) for _,t in docs]
    ours=BM25(docs)
    ref=reference.BM25Okapi(toks,k1=1.5,b=.75)
    ref.idf=dict(ours.idf)  # Assignment's positive-IDF formula
    for q in ['apple','banana','pear','apple pear','unknown']:
        scores=ref.get_scores(list(set(bm25_tokens(q))))
        ranked_ref=[docs[i][0] for i in sorted(
            (i for i,s in enumerate(scores) if s>0),
            key=lambda i:(-scores[i], docs[i][0]))]
        assert ours.search(q,k=10)==ranked_ref
