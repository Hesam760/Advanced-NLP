from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import yaml

SYSTEMS = (
    ('dev', 'dense_fair'), ('dev', 'hybrid'), ('hidden', 'hybrid'),
    ('dev', 'title'), ('dev', 'sentence'), ('dev', 'masked')
)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def safe_extract(archive: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination.resolve()) or item.filename.startswith('/'):
                raise ValueError(f'Unsafe ZIP member: {item.filename}')
        z.extractall(destination)


def find_saved_runs(roots, target):
    """Find extracted phase-2 outputs or safely unpack the exported ZIP."""
    required = ['runs/dev/hybrid/predictions.jsonl', 'runs/dev/masked/run_config.yaml']
    target = Path(target)
    for root in map(Path, roots):
        if not root.exists():
            continue
        if all((root / r).is_file() for r in required):
            return root
        for item in root.rglob('runs/dev/hybrid/predictions.jsonl'):
            base = item.parents[3]  # .../<root>/runs/dev/hybrid/predictions.jsonl
            if all((base / r).is_file() for r in required):
                return base
    for root in map(Path, roots):
        if not root.exists():
            continue
        archives = ([root] if root.is_file() and root.suffix.lower()=='.zip'
                    else sorted(root.rglob('*.zip'))) 
        for archive in archives:
            if not zipfile.is_zipfile(archive):
                continue
            with zipfile.ZipFile(archive) as z:
                if all(r in z.namelist() for r in required):
                    safe_extract(archive, target)
                    return target
    raise FileNotFoundError('Phase 2 runs/dev/hybrid and runs/dev/masked not found in input ZIP/dataset.')


def find_labels(roots, target):
    rel = 'course_pack_v1/labels_dev.jsonl'
    target = Path(target)
    for root in map(Path, roots):
        if not root.exists():
            continue
        if (root / rel).is_file():
            return root / rel
        for path in root.rglob('labels_dev.jsonl'):
            if path.parent.name == 'course_pack_v1':
                return path
    for root in map(Path, roots):
        if not root.exists():
            continue
        archives = ([root] if root.is_file() and root.suffix.lower()=='.zip'
                    else sorted(root.rglob('*.zip')))
        for archive in archives:
            if not zipfile.is_zipfile(archive):
                continue
            with zipfile.ZipFile(archive) as z:
                if rel in z.namelist():
                    safe_extract(archive,target)
                    return target / rel
    raise FileNotFoundError('course_pack_v1/labels_dev.jsonl was not found. Add Phase2_Kaggle_Input.zip.')


def inspect_saved_run(root, split, system, labels=None, corpus_ids=None):
    root=Path(root)
    folder=root/'runs'/split/system
    cfg=yaml.safe_load((folder/'run_config.yaml').read_text(encoding='utf-8'))
    rows=jsonl(folder/'predictions.jsonl')
    assert cfg['phase']==2 and cfg['split']==split, (folder, 'configuration metadata mismatch')
    qids=[r['qid'] for r in rows]
    assert len(qids)==len(set(qids)), (folder, 'duplicate qid')
    assert len(rows)==300, (folder, 'expected full 300-question run')
    if labels is not None and split=='dev':
        assert set(qids)==set(labels), (folder, 'wrong qids')
    failures=[]
    n_abstained=0
    for row in rows:
        retrieved=row['retrieved']
        cited=row.get('cited',[])
        answer=row['answer']
        abstained=row.get('abstained',False)
        if len(retrieved)!=len(set(retrieved)) or len(cited)!=len(set(cited)):
            failures.append((row['qid'],'duplicates'))
        if not set(cited).issubset(retrieved):
            failures.append((row['qid'],'citation not in retrieved'))
        if corpus_ids is not None and not set(retrieved).issubset(corpus_ids):
            failures.append((row['qid'],'unknown document'))
        if abstained:
            n_abstained+=1
            if answer!='' or cited:
                failures.append((row['qid'],'invalid abstention'))
        elif not isinstance(answer,str) or not answer.strip():
            failures.append((row['qid'],'empty non-abstained answer'))
        if system=='masked' and labels is not None:
            if set(retrieved).intersection(labels[row['qid']]['gold_doc_ids']):
                failures.append((row['qid'],'gold leak'))
    assert not failures, failures[:10]
    if system=='masked':
        assert cfg['settings']['retrieval'].get('gold_masked_before_retrieval') is True
    return {'n':len(rows),'abstained':n_abstained,'config':cfg,'predictions':folder/'predictions.jsonl','config_path':folder/'run_config.yaml'}


_METRIC = re.compile(r'^\s{2,}([a-z_0-9@]+)\s+([-+]?\d+(?:\.\d+)?)\s+\[', re.I)
_TYPE = re.compile(r'^type:\s*([a-z_]+)\s+\(n=(\d+)\)')

def parse_course_report(text):
    """Return actual official score.py metrics; don't invent hidden scores."""
    result={'overall':{}, 'types':{}, 'paired':{}}
    dest=None
    for line in text.splitlines():
        stripped=line.strip()
        if stripped.startswith('overall ('):
            dest=result['overall']
        elif _TYPE.match(stripped):
            name=_TYPE.match(stripped).group(1)
            result['types'][name]={}
            dest=result['types'][name]
        elif stripped.startswith('paired difference:'):
            dest=result['paired']
        elif stripped.startswith('valid:') or stripped.startswith('Scoring'):
            pass
        m=_METRIC.match(line)
        if m and dest is not None:
            dest[m.group(1)]=float(m.group(2))
    return result


def report_for(system,split):
    if split=='hidden':
        return 'full_hidden_hybrid_hidden.txt'
    return {'dense_fair':'full_dev_dense_fair.txt',
            'hybrid':'full_dev_hybrid_minus_dense_fair.txt',
            'title':'full_dev_title.txt',
            'sentence':'full_dev_sentence.txt',
            'masked':'full_dev_masked.txt'}[system]


def make_master_config(audits, source_zip=None):
    """Derive one master YAML from real run_config records; fail if fairness differs."""
    dense=audits[('dev','dense_fair')]['config']
    hybrid=audits[('dev','hybrid')]['config']
    llm_d=dense['models']['llm']; llm_h=hybrid['models']['llm']
    assert llm_d==llm_h, 'Dense and Hybrid LLM configs differ'
    for field in ('max_context_tokens','max_input_tokens','max_new_tokens',
                  'embedding_max_length','prompt_text','enable_thinking','context_template'):
        assert dense['settings'][field]==hybrid['settings'][field], field
    assert llm_h['do_sample'] is False, 'Generation must be greedy'
    assert hybrid['settings']['retrieval']['rrf_k']==60
    settings=dense['settings']
    master={
        'phase':2,
        'models':dense['models'],
        'seed':llm_d.get('seed',42),
        'generation':{
            'do_sample':llm_d['do_sample'],
            'temperature':0,
            'max_new_tokens':settings['max_new_tokens'],
            'max_input_tokens':settings['max_input_tokens'],
            'max_context_tokens':settings['max_context_tokens'],
            'enable_thinking':settings['enable_thinking'],
            'prompt_text':settings['prompt_text'],
            'context_template':settings['context_template'],
        },
        'embedding':{'max_length':settings['embedding_max_length']},
        'retrieval':{
            'final_k':hybrid['settings']['retrieval']['retrieved_k'],
            'bm25_k1':hybrid['settings']['retrieval']['bm25_k1'],
            'bm25_b':hybrid['settings']['retrieval']['bm25_b'],
            'rrf_k':hybrid['settings']['retrieval']['rrf_k'],
            'dense_candidates':hybrid['settings']['retrieval']['dense_top_k'],
            'bm25_candidates':hybrid['settings']['retrieval']['bm25_top_k'],
        },
        'runs': {f'{split}_{system}':audits[(split,system)]['config']['settings']['retrieval']
                 for split,system in SYSTEMS},
        'source':'Generated from actual phase-2 run_config.yaml files',
    }
    if source_zip and Path(source_zip).is_file():
        master['source_zip_sha256']=hashlib.sha256(Path(source_zip).read_bytes()).hexdigest()
    return master


def write_json(path, content):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def write_yaml(path, content):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(yaml.safe_dump(content,sort_keys=False,allow_unicode=True),encoding='utf-8')
