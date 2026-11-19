import json
from pathlib import Path

import pytest
import yaml

from src.phase2_engineering import (find_saved_runs, inspect_saved_run, parse_course_report,
                                   safe_extract, write_yaml)


def test_scoring_report_parser_by_question_type():
    report='''overall (n=300)
  em                44.7   [ 39.3,  50.3]
  hitall@10         53.0   [ 47.3,  58.3]
type: bridge_comparison (n=66)
  em                47.0   [ 34.8,  59.1]
paired difference: foo minus bar (points, 95 % bootstrap interval)
  em                -3.7   [ -7.7,  +0.3]
'''
    parsed=parse_course_report(report)
    assert parsed['overall']['em']==44.7
    assert parsed['overall']['hitall@10']==53.0
    assert parsed['types']['bridge_comparison']['em']==47.0
    assert parsed['paired']['em']==-3.7


def test_full_masked_run_audit(tmp_path):
    root=tmp_path/'result'
    folder=root/'runs'/'dev'/'masked'
    folder.mkdir(parents=True)
    cfg={'phase':2,'split':'dev','settings':{'retrieval':{
        'gold_masked_before_retrieval':True,'method':'hybrid_rrf'}}}
    (folder/'run_config.yaml').write_text(yaml.safe_dump(cfg))
    labels={}
    with (folder/'predictions.jsonl').open('w') as f:
        for i in range(300):
            qid=f'q{i:04d}'
            labels[qid]={'gold_doc_ids':['gold']}
            row={'qid':qid,'answer':'','retrieved':['other'],'cited':[], 'abstained':True}
            f.write(json.dumps(row)+'\n')
    actual=inspect_saved_run(root,'dev','masked', labels=labels,corpus_ids={'gold','other'})
    assert actual['n']==300 and actual['abstained']==300
    with (folder/'predictions.jsonl').open('a') as f:
        f.write(json.dumps({'qid':'q0000','answer':'bad','retrieved':['gold'],'cited':[], 'abstained':False})+'\n')
    with pytest.raises(AssertionError):
        inspect_saved_run(root,'dev','masked',labels=labels)


def test_write_yaml(tmp_path):
    p=tmp_path/'config.yaml'
    write_yaml(p,{'phase':2,'models':{'llm':'qwen'}})
    assert yaml.safe_load(p.read_text())['phase']==2
