"""Finalize A1 config-driven notebook without executing GPU models.

Supports notebooks with additional/removed unrelated cells by matching
original code-cell fingerprints; refuses modified target cells rather than
applying risky position-based edits.

Usage (from repository root):
    python finalize_a1.py

Requires PyYAML; uses only stdlib + PyYAML. Keeps original notebook and predictions unchanged.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
from pathlib import Path

import yaml

ROOT = Path.cwd().resolve()
ORIGINAL = ROOT / "advanced-nlp (1).ipynb"
NEW_NOTEBOOK = ROOT / "advanced-nlp-config-driven.ipynb"
SYSTEMS = ("closed_book", "oracle", "dense")

# Fingerprints of the 26 original CODE cells that must be patched.
# Additional/deleted/reordered unrelated cells are tolerated; modified target
# cells are NOT silently patched because their semantics may have changed.
EXPECTED_CELL_HASHES = {8: 'f6895581e613e4b0967843208da164fb71927bff0d0a7a489b5ee0c318edebb1', 9: 'f8ba30ac10bc880ca6daec584941492e42d09543d294cd896fcc45f2cba5eeda', 10: '3b8a53728328734d1103a3e66daff34b23deb6f8adab5da3dfcb228686cff7e0', 16: 'b50f2232baf050d7bb7ab310e2efc5d9d757b19bbadc14ffcc4e01a308d86722', 17: '11bf26dd0c40f31532aa14f6aadfbe7f9cec8f22ff7498f242e1052fb3599682', 18: '3716989098cfedf9cfe8096dee94525282806bf057caceff3cad16913346c07d', 23: 'bb4df26fa246eaa2c03f7d67aa525d7b0fe55ca1aceef616bf67c49552b89bcb', 24: 'e3c150170c1217ec1f5fc9deb35ea12ede7eea341d377626b2085d0526e4e8d1', 26: '0aa69e4b81b560a3e796184c6cc8a7c81d9e182d6c6ba9e3455f8421caf3a6fe', 30: 'ec4c998be813188fcc0fe628f1d127b826827d6fbc920f8b34ff752da8833da2', 31: '8c4d3d0f9c55c0b2eeca3ec58788aec3c68242f64ccd039ce6b14f5a43a37d71', 32: '21a28a205af78eac1d62989c9c02df70802a2f6481e377332d0057136d240bec', 33: '4ccb4f95d3348a65b9fa459df559ab02fb4c0839d81edadf7a04116ce3a514e4', 38: '931b4de3232a1aa55a1e53230d384f1caaee20c6cd9c7dd36d38e4ed6ce96007', 39: '85bca62df4a144e711b677bb04788cb53381fccc3a92209b95abdcb832b769be', 40: 'd07138b2625f1cf43cb26fc25e40806568d935044e995a47f960dd3583441999', 41: '1a91d94d1faaa21067a1eaa8f7c938d6d000bf8fda950025065102e9e3847ac1', 42: '1492499c0d322450c4beded99e09ff5b3db59961da7e80543a400f4405955398', 43: '57d23fa262485d707c99abd95001aa47b0babfd07718c7adecee256e7b9925f2', 46: '168a42e7054b5fe5645171a294b3274d7d8e9152ada1142778bc43246fe193ab', 48: 'b5c5d739a5c30a6413f9a1e5e935862a27d8e3123c851b29950084675982ebbb', 49: '9215319a0d3f0a9651e7118a93946d73ffe3b7e4f0ca056b5b50309e00c6aa0f', 50: '1d4517d5498ec5198b6bca3cbe875ebc362a920c1726e7360a7ce0fd43747a87', 52: 'f464abe5fa863ed2ab15ac0a0211cdb639b767dd94ef73abf5f6dd488dfe64a9', 54: '405fd9c73214dca098ca4cd02857d6d9cdf0312fbb22fe2a503ed5e13b4c05ca', 55: '23e846e814dc9bd5d0f64348e3062377ebb7e5996e6ce4bcea3bcead88028291'}
CELL_MAP = {}


def map_target_cells(nb):
    by_hash = {}
    for actual_i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        code = "".join(cell.get("source", []))
        digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
        by_hash.setdefault(digest, []).append(actual_i)

    missing = []
    mapping = {}
    for canonical_i, digest in EXPECTED_CELL_HASHES.items():
        found = by_hash.get(digest, [])
        if len(found) == 1:
            mapping[canonical_i] = found[0]
        else:
            missing.append((canonical_i, len(found)))
    if missing:
        details = ", ".join(f"original cell {i}: {n} exact matches" for i,n in missing)
        raise SystemExit(
            f"Notebook is not the same code version as the uploaded reference. "
            f"Could not safely match: {details}. No changes made.\n"
            "Do not remove the safety check; send the diagnostic output and we will adapt the patches."
        )
    indexes = [mapping[i] for i in sorted(mapping)]
    if indexes != sorted(indexes):
        raise SystemExit("Target cells were reordered. No changes made; review notebook first.")
    return mapping



def replace_one(source: str, old: str, new: str, cell: int) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"Cell {cell}: expected one occurrence of {old!r}, found {count}")
    return source.replace(old, new, 1)


def replace_all(source: str, old: str, new: str, cell: int, min_count=1) -> str:
    count = source.count(old)
    if count < min_count:
        raise ValueError(f"Cell {cell}: expected >= {min_count} of {old!r}, found {count}")
    return source.replace(old, new)


def prompt_literal(source: str, var: str, cell: int):
    expr = rf'(?m)^{re.escape(var)}\s*=\s*(""".*?""")(?:\.strip\(\))?'
    matches = list(re.finditer(expr, source, flags=re.S))
    if len(matches) != 1:
        raise ValueError(f"Cell {cell}: cannot find prompt assignment for {var}")
    m = matches[0]
    text = ast.literal_eval(m.group(1)).strip()
    return text, m


def replace_prompt(source: str, var: str, cell: int, cfg_var: str):
    _, m = prompt_literal(source, var, cell)
    return source[:m.start()] + f"{var} = a1_prompt({cfg_var})" + source[m.end():]


def get_cell(nb, i):
    return "".join(nb["cells"][CELL_MAP[i]]["source"])


def put_cell(nb, i, code):
    nb["cells"][CELL_MAP[i]]["source"] = code.splitlines(keepends=True)
    # Deliberately do not preserve execution outputs in the NEW notebook.
    if nb["cells"][CELL_MAP[i]]["cell_type"] == "code":
        nb["cells"][CELL_MAP[i]]["execution_count"] = None
        nb["cells"][CELL_MAP[i]]["outputs"] = []


def main():
    if not ORIGINAL.exists():
        raise SystemExit(f"Missing original notebook: {ORIGINAL}")
    if NEW_NOTEBOOK.exists():
        raise SystemExit(f"Output exists (will not overwrite): {NEW_NOTEBOOK}")
    nb = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    global CELL_MAP
    CELL_MAP = map_target_cells(nb)
    print(f"Matched {len(CELL_MAP)} original code cells safely in a {len(nb['cells'])}-cell notebook.")

    configs = {}
    for system in SYSTEMS:
        path = ROOT / "configs" / (system + ".yaml")
        if not path.is_file():
            raise SystemExit(f"Missing {path}")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or "settings" not in config or "models" not in config:
            raise SystemExit(f"Invalid configuration: {path}")
        configs[system] = config

    if not (ROOT / "src" / "config.py").exists() or not (ROOT / "src" / "retrieval.py").exists():
        raise SystemExit("Missing src/config.py or src/retrieval.py")
    if not (ROOT / "src" / "metrics.py").exists():
        raise SystemExit("Missing src/metrics.py")

    def enrich(system):
        cfg = configs[system]
        llm = cfg["models"]["llm"]
        llm.setdefault("load_in_4bit", True)
        llm.setdefault("quantization", "nf4")
        llm.setdefault("double_quantization", True)
        llm.setdefault("compute_dtype", "float16")
        llm.setdefault("model_dtype", "float16")
        llm.setdefault("attention_implementation", "sdpa")
        llm.setdefault("device_map", "auto")
        llm.setdefault("low_cpu_mem_usage", True)
        gen = cfg["settings"].setdefault("generation", {})
        gen.setdefault("do_sample", False)
        gen.setdefault("enable_thinking", False)
        gen.setdefault("max_new_tokens", 64)
        gen.setdefault("device", "cuda:0")
        cfg["settings"]["prompt_file"] = f"prompts/{system}.txt"
        if system != "closed_book":
            gen.setdefault("max_input_tokens", 6144)
        if system == "dense":
            retrieval = cfg["settings"].setdefault("retrieval", {})
            retrieval.setdefault("k", 10)
            retrieval.setdefault("document_mask", True)
            retrieval.setdefault("unit", "passage")
            retrieval.setdefault("method", "exact_cosine")
            retrieval.setdefault("normalize_embeddings", True)
            retrieval.setdefault("chroma_batch_size", 64)
            retrieval.setdefault("chroma_dir", "/kaggle/working/chroma_qwen3_embedding4b")
            retrieval.setdefault("collection_name", "course_passages_qwen3_emb4b_v1")
            retrieval.setdefault("space", "cosine")
            emb = cfg["settings"].setdefault("embedding", {})
            emb.setdefault("max_seq_length", 2048)
            emb.setdefault("exploratory_batch_size", 8)
            emb.setdefault("document_batch_size", retrieval.pop("document_batch_size", 4))
            emb.setdefault("query_batch_size", 1)
            emb.setdefault("query_instruction", "Given a question, retrieve passages that contain the information needed to answer the question")
            emb.setdefault("precision", "float32")
            emb.setdefault("device", "cuda:0")
            emb.setdefault("torch_dtype", "float16")
            cfg.setdefault("retrieval_predictions", {
                "dev": "runs/dev/dense_chroma_qwen3_4b/predictions.jsonl",
                "hidden": "runs/hidden/dense_chroma_qwen3_4b/predictions.jsonl",
            })
            gen.setdefault("context_top_k", 10)
            gen.setdefault("context_truncation", "remove_lowest_ranked_passage")

    for s in SYSTEMS:
        enrich(s)

    originals = {i: get_cell(nb, i) for i in EXPECTED_CELL_HASHES}
    for i, key, system in [(23, "CLOSED_BOOK_SYSTEM_PROMPT", "closed_book"),
                            (30, "ORACLE_SYSTEM_PROMPT", "oracle"),
                            (48, "SYSTEM_PROMPT", "dense")]:
        prompt, _ = prompt_literal(originals[i], key, i)
        configs[system]["settings"]["prompt_file"] = f"prompts/{system}.txt"
        prompt_file = ROOT / configs[system]["settings"]["prompt_file"]
        if prompt_file.exists() and prompt_file.read_text(encoding="utf-8").strip() != prompt:
            raise SystemExit(f"Existing prompt differs: {prompt_file}. Review before overwriting.")

    # Ensure the notebook will not need GPU when merely loading YAML.
    bootstrap = '''# A1: shared run configuration; this cell does NOT load any model.
from pathlib import Path
import os
import sys


def locate_a1_project():
    options = []
    if os.environ.get("A1_PROJECT_ROOT"):
        options.append(Path(os.environ["A1_PROJECT_ROOT"]))
    options.extend([Path.cwd(), *Path.cwd().parents])
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        # Add the private project source/configs as a Kaggle Input dataset for replay.
        options.extend(p.parent.parent for p in kaggle_input.rglob("configs/dense.yaml"))
    for root in options:
        if (root / "src" / "config.py").is_file() and all(
            (root / "configs" / f"{name}.yaml").is_file()
            for name in ("closed_book", "oracle", "dense")
        ):
            return root.resolve()
    raise FileNotFoundError(
        "A1 project files not found. Set A1_PROJECT_ROOT or attach "
        "the repository files as a Kaggle Input dataset."
    )

A1_ROOT = locate_a1_project()
if str(A1_ROOT) not in sys.path:
    sys.path.insert(0, str(A1_ROOT))

from src.config import load_config
CLOSED_CFG = load_config(A1_ROOT / "configs/closed_book.yaml")
ORACLE_CFG = load_config(A1_ROOT / "configs/oracle.yaml")
DENSE_CFG = load_config(A1_ROOT / "configs/dense.yaml")
DENSE_K = int(DENSE_CFG["settings"]["retrieval"]["k"])
A1_OUTPUT_ROOT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else A1_ROOT


def a1_prediction_path(cfg, split):
    return A1_OUTPUT_ROOT / cfg["source_predictions"][split]


def a1_retrieval_path(split):
    return A1_OUTPUT_ROOT / DENSE_CFG["retrieval_predictions"][split]

assert CLOSED_CFG["models"]["llm"]["name"] == ORACLE_CFG["models"]["llm"]["name"]
assert ORACLE_CFG["models"]["llm"]["name"] == DENSE_CFG["models"]["llm"]["name"]


def a1_prompt(cfg):
    path = A1_ROOT / cfg["settings"]["prompt_file"]
    return path.read_text(encoding="utf-8").strip()


def a1_torch_dtype(torch, cfg):
    name = cfg["models"]["llm"]["compute_dtype"]
    if name not in {"float16", "bfloat16", "float32"}:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return getattr(torch, name)

print("A1 loaded three YAML configurations from:", A1_ROOT)
'''

    # Cell 8: exploratory Qwen embedding baseline, same embedder as final run.
    s = originals[8]
    s = replace_one(s, 'EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-4B"', 'EMBED_MODEL_ID = DENSE_CFG["models"]["embedder"]', 8)
    s = replace_one(s, 'model.max_seq_length = 2048', 'model.max_seq_length = DENSE_CFG["settings"]["embedding"]["max_seq_length"]', 8)
    s = replace_all(s, 'batch_size=8', 'batch_size=DENSE_CFG["settings"]["embedding"]["exploratory_batch_size"]', 8, 2)
    put_cell(nb, 8, s)

    # Cell 9: prior experimental exact cosine retriever; its default k is YAML-managed.
    s = originals[9]
    s = replace_one(s, 'def dense_search(query_index, k=10, exclude_ids=None):', 'def dense_search(query_index, k=None, exclude_ids=None):\n    k = DENSE_K if k is None else int(k)', 9)
    s = replace_one(s, 'dense_search(i, k=10)', 'dense_search(i, k=DENSE_K)', 9)
    put_cell(nb, 9, s)

    # Cell 10: use the actual module covered by pytest, not a second Notebook copy.
    s = originals[10]
    match = re.search(r'(?m)^def retrieval_metrics\(ranked, gold_ids\):.*?(?=^# Evaluate all questions)', s, flags=re.S)
    if not match:
        raise SystemExit('Cannot find metrics definition in cell 10')
    s = s[:match.start()] + 'from src.metrics import retrieval_metrics\n\n' + s[match.end():]
    put_cell(nb, 10, s)

    # Shared LLM loading for experimental Oracle + other Phase-1 systems.
    for i, cfgvar in [(16, "ORACLE_CFG"), (49, "DENSE_CFG"), (54, "DENSE_CFG")]:
        s = originals[i]
        s = replace_one(s, 'MODEL_ID = "Qwen/Qwen3.5-9B"', f'MODEL_ID = {cfgvar}["models"]["llm"]["name"]', i)
        s = replace_one(s, 'load_in_4bit=True', f'load_in_4bit={cfgvar}["models"]["llm"]["load_in_4bit"]', i)
        s = replace_one(s, 'bnb_4bit_quant_type="nf4"', f'bnb_4bit_quant_type={cfgvar}["models"]["llm"]["quantization"]', i)
        s = replace_one(s, 'bnb_4bit_use_double_quant=True', f'bnb_4bit_use_double_quant={cfgvar}["models"]["llm"]["double_quantization"]', i)
        s = replace_one(s, 'bnb_4bit_compute_dtype=torch.float16', f'bnb_4bit_compute_dtype=a1_torch_dtype(torch, {cfgvar})', i)
        s = replace_one(s, 'dtype=torch.float16', f'dtype=getattr(torch, {cfgvar}["models"]["llm"]["model_dtype"])', i)
        s = replace_one(s, 'device_map="auto"', f'device_map={cfgvar}["models"]["llm"]["device_map"]', i)
        s = replace_one(s, 'attn_implementation="sdpa"', f'attn_implementation={cfgvar}["models"]["llm"]["attention_implementation"]', i)
        s = replace_one(s, 'low_cpu_mem_usage=True', f'low_cpu_mem_usage={cfgvar}["models"]["llm"]["low_cpu_mem_usage"]', i)
        put_cell(nb, i, s)

    # Prior Oracle experiment and final Oracle use the saved exact prompt.
    s = originals[17]
    put_cell(nb, 17, replace_prompt(s, "SYSTEM_PROMPT", 17, "ORACLE_CFG"))
    s = originals[18]
    s = replace_one(s, 'def generate_oracle_answer(question, max_new_tokens=64):',
                    'def generate_oracle_answer(question, max_new_tokens=None):\n    if max_new_tokens is None:\n        max_new_tokens = ORACLE_CFG["settings"]["generation"]["max_new_tokens"]',18)
    s = replace_one(s, 'enable_thinking=False', 'enable_thinking=ORACLE_CFG["settings"]["generation"]["enable_thinking"]',18)
    s = replace_one(s, 'if input_length > 6144:', 'if input_length > ORACLE_CFG["settings"]["generation"]["max_input_tokens"]:',18)
    s = replace_one(s, 'do_sample=False', 'do_sample=ORACLE_CFG["settings"]["generation"]["do_sample"]',18)
    put_cell(nb,18,s)

    s = replace_prompt(originals[23], 'CLOSED_BOOK_SYSTEM_PROMPT',23,'CLOSED_CFG')
    s = replace_one(s, 'def generate_closed_book_answer(question, max_new_tokens=64):',
        'def generate_closed_book_answer(question, max_new_tokens=None):\n    if max_new_tokens is None:\n        max_new_tokens = CLOSED_CFG["settings"]["generation"]["max_new_tokens"]',23)
    s = replace_one(s, 'enable_thinking=False', 'enable_thinking=CLOSED_CFG["settings"]["generation"]["enable_thinking"]',23)
    s = replace_one(s, 'do_sample=False', 'do_sample=CLOSED_CFG["settings"]["generation"]["do_sample"]',23)
    put_cell(nb,23,s)

    put_cell(nb,30,replace_prompt(originals[30], 'ORACLE_SYSTEM_PROMPT',30,'ORACLE_CFG'))
    s = replace_one(originals[31], 'MAX_INPUT_TOKENS = 6144', 'MAX_INPUT_TOKENS = ORACLE_CFG["settings"]["generation"]["max_input_tokens"]',31)
    s = replace_one(s, 'enable_thinking=False','enable_thinking=ORACLE_CFG["settings"]["generation"]["enable_thinking"]',31)
    put_cell(nb,31,s)
    s = replace_one(originals[32], 'def generate_oracle_answer(question, max_new_tokens=64):',
        'def generate_oracle_answer(question, max_new_tokens=None):\n    if max_new_tokens is None:\n        max_new_tokens = ORACLE_CFG["settings"]["generation"]["max_new_tokens"]',32)
    s = replace_one(s,'do_sample=False','do_sample=ORACLE_CFG["settings"]["generation"]["do_sample"]',32)
    put_cell(nb,32,s)

    # Final dense embedding and Chroma setup.
    s = replace_one(originals[38], 'EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-4B"', 'EMBED_MODEL_ID = DENSE_CFG["models"]["embedder"]',38)
    s = replace_one(s, 'device="cuda:0"', 'device=DENSE_CFG["settings"]["embedding"]["device"]',38)
    put_cell(nb,38,s)
    s = originals[39]
    s = replace_one(s, '"/kaggle/working/chroma_qwen3_embedding4b"', 'DENSE_CFG["settings"]["retrieval"]["chroma_dir"]',39)
    s = replace_one(s, 'COLLECTION_NAME = "course_passages_qwen3_emb4b_v1"', 'COLLECTION_NAME = DENSE_CFG["settings"]["retrieval"]["collection_name"]',39)
    s = replace_one(s, '"hnsw:space": "cosine"', '"hnsw:space": DENSE_CFG["settings"]["retrieval"]["space"]',39)
    s = replace_one(s, 'batch_size=4', 'batch_size=DENSE_CFG["settings"]["embedding"]["document_batch_size"]',39)
    s = replace_one(s, 'CHROMA_BATCH_SIZE = 64', 'CHROMA_BATCH_SIZE = DENSE_CFG["settings"]["retrieval"]["chroma_batch_size"]',39)
    put_cell(nb,39,s)
    s = originals[40]
    m = re.search(r'(?m)^QUERY_INSTRUCTION\s*=\s*\(.*?\n\)',s,flags=re.S)
    if not m: raise SystemExit('Cannot find QUERY_INSTRUCTION in cell 40')
    s = s[:m.start()] + 'QUERY_INSTRUCTION = DENSE_CFG["settings"]["embedding"]["query_instruction"]' + s[m.end():]
    s = replace_one(s, 'batch_size=1', 'batch_size=DENSE_CFG["settings"]["embedding"]["query_batch_size"]',40)
    put_cell(nb,40,s)

    # Swap Notebook dense retrieval implementation for the pytest-tested module.
    # The same Chroma stored vectors and the same top-k exact cosine are used.
    put_cell(nb,41,'''from src.retrieval import exact_cosine_search


def dense_search(query, k=None, exclude_ids=frozenset()) -> list[str]:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if k is None:
        k = DENSE_K
    if not DENSE_CFG["settings"]["retrieval"]["document_mask"] and exclude_ids:
        raise ValueError("Document mask disabled in config; cannot exclude IDs")
    query_embedding = embed_query(query)
    stored = collection.get(include=["embeddings"], limit=collection.count())
    return exact_cosine_search(
        doc_ids=stored["ids"],
        document_embeddings=stored["embeddings"],
        query_embedding=query_embedding,
        k=k,
        exclude_ids=exclude_ids,
    )
''')

    s = originals[42]
    s = replace_one(s,'dense_search(test_query, k=10)','dense_search(test_query, k=DENSE_K)',42)
    s = replace_all(s,'== 10','== DENSE_K',42,2)
    put_cell(nb,42,s)
    s = replace_one(originals[43], 'k=10', 'k=DENSE_K',43)
    put_cell(nb,43,s)
    s = originals[46]
    s = replace_one(s, 'len(row["retrieved"]) == 10', 'len(row["retrieved"]) == DENSE_K',46)
    s = replace_one(s, 'print("Retrieved documents per question:", 10)', 'print("Retrieved documents per question:", DENSE_K)',46)
    put_cell(nb,46,s)

    s = replace_prompt(originals[48], 'SYSTEM_PROMPT',48,'DENSE_CFG')
    s = replace_one(s, 'MAX_INPUT_TOKENS = 6144', 'MAX_INPUT_TOKENS = DENSE_CFG["settings"]["generation"]["max_input_tokens"]',48)
    s = replace_one(s, 'for n in range(len(doc_ids), 0, -1):',
                   'for n in range(min(len(doc_ids), DENSE_CFG["settings"]["generation"]["context_top_k"]), 0, -1):',48)
    s = replace_one(s, 'enable_thinking=False', 'enable_thinking=DENSE_CFG["settings"]["generation"]["enable_thinking"]',48)
    s = replace_one(s,'max_new_tokens=64\n):', 'max_new_tokens=None\n):\n\n    if max_new_tokens is None:\n        max_new_tokens = DENSE_CFG["settings"]["generation"]["max_new_tokens"]',48)
    s = replace_one(s, 'do_sample=False', 'do_sample=DENSE_CFG["settings"]["generation"]["do_sample"]',48)
    put_cell(nb,48,s)

    s = originals[50]
    s = replace_all(s, '== 10','== DENSE_K',50,2)
    s = replace_one(s, 'max_new_tokens=64', 'max_new_tokens=DENSE_CFG["settings"]["generation"]["max_new_tokens"]',50)
    put_cell(nb,50,s)

    s = originals[52]
    s = replace_all(s, '== 10','== DENSE_K',52,2)
    s = replace_all(s, '!= 10','!= DENSE_K',52,2)
    s = replace_one(s, 'k=10', 'k=DENSE_K',52)
    put_cell(nb,52,s)
    s = replace_one(originals[55], 'max_new_tokens=64', 'max_new_tokens=DENSE_CFG["settings"]["generation"]["max_new_tokens"]',55)
    put_cell(nb,55,s)

    # All output locations are config-derived (Kaggle working dir or local project root).
    def replace_output_path(cell, absolute_path, expression):
        source = get_cell(nb, cell)
        pattern = r'Path\(\s*"' + re.escape(absolute_path) + r'"\s*\)'
        new_source, count = re.subn(pattern, expression, source)
        if count != 1:
            raise ValueError(f"Cell {cell}: expected path {absolute_path}; found {count}")
        put_cell(nb, cell, new_source)

    for i, path, expression in [
        (24, "/kaggle/working/runs/dev/closed_book", 'a1_prediction_path(CLOSED_CFG, "dev").parent'),
        (26, "/kaggle/working/runs/hidden/closed_book", 'a1_prediction_path(CLOSED_CFG, "hidden").parent'),
        (33, "/kaggle/working/runs/dev/context_oracle_qwen3_5_9b", 'a1_prediction_path(ORACLE_CFG, "dev").parent'),
        (43, "/kaggle/working/runs/dev/dense_chroma_qwen3_4b", 'a1_retrieval_path("dev").parent'),
        (46, "/kaggle/working/runs/dev/dense_chroma_qwen3_4b/predictions.jsonl", 'a1_retrieval_path("dev")'),
        (50, "/kaggle/working/runs/dev/dense_chroma_qwen3_4b/predictions.jsonl", 'a1_retrieval_path("dev")'),
        (50, "/kaggle/working/runs/dev/dense_rag_qwen3_5_9b", 'a1_prediction_path(DENSE_CFG, "dev").parent'),
        (52, "/kaggle/working/runs/hidden/dense_chroma_qwen3_4b", 'a1_retrieval_path("hidden").parent'),
        (55, "/kaggle/working/runs/hidden/dense_rag_qwen3_5_9b", 'a1_prediction_path(DENSE_CFG, "hidden").parent'),
    ]:
        replace_output_path(i, path, expression)

    # Execution device and embedding dtype are part of the YAML, not magic constants.
    for i, cfg_name in [(18, "ORACLE_CFG"), (23, "CLOSED_CFG"), (32, "ORACLE_CFG")]:
        src = get_cell(nb, i)
        src = replace_one(src, 'inputs = inputs.to("cuda:0")',
                          f'inputs = inputs.to({cfg_name}["settings"]["generation"]["device"])',i)
        put_cell(nb,i,src)
    s = get_cell(nb,48)
    s = replace_one(s, 'value.to("cuda:0")',
                    'value.to(DENSE_CFG["settings"]["generation"]["device"])',48)
    put_cell(nb,48,s)
    for i in (8,38):
        src = get_cell(nb,i)
        if i == 8:
            src = 'import torch\n' + src
        src = replace_one(src, '"torch_dtype": torch.float16',
                '"torch_dtype": getattr(torch, DENSE_CFG["settings"]["embedding"]["torch_dtype"])', i)
        put_cell(nb,i,src)

    # Keep cosine normalization and precision under the same Dense YAML.
    for i in (8,39,40):
        src = get_cell(nb,i)
        src = replace_all(src, 'normalize_embeddings=True',
                          'normalize_embeddings=DENSE_CFG["settings"]["retrieval"]["normalize_embeddings"]',i)
        if i in (39,40):
            src = replace_one(src,'precision="float32"',
                  'precision=DENSE_CFG["settings"]["embedding"]["precision"]',i)
        put_cell(nb,i,src)

    # Notebook's demonstration mask check must also honor configurable k.
    src = get_cell(nb,42)
    src = replace_one(src,'    k=10,','    k=DENSE_K,',42)
    src = replace_one(src,'masked_10[:9] == ranked_10[1:10]',
                      'masked_10[:DENSE_K-1] == ranked_10[1:DENSE_K]',42)
    put_cell(nb,42,src)

    # Compile-check every altered Python cell (not IPython %pip magic cells).
    edited = [i for i in EXPECTED_CELL_HASHES if get_cell(nb, i) != originals[i]]
    for i in edited:
        ast.parse(get_cell(nb,i), filename=f"Notebook cell {i}")
    ast.parse(bootstrap, filename="A1 bootstrap cell")

    # Only now write files: all notebook patches above have been validated.
    (ROOT / "prompts").mkdir(exist_ok=True)
    for i,key,system in [(23,"CLOSED_BOOK_SYSTEM_PROMPT","closed_book"),(30,"ORACLE_SYSTEM_PROMPT","oracle"),(48,"SYSTEM_PROMPT","dense")]:
        original_prompt,_ = prompt_literal(originals[i],key,i)
        p = ROOT / "prompts" / f"{system}.txt"
        if not p.exists():
            p.write_text(original_prompt + "\n",encoding="utf-8")

    for system in SYSTEMS:
        p = ROOT / "configs" / (system + ".yaml")
        original = p.read_text(encoding="utf-8")
        new = yaml.safe_dump(configs[system], allow_unicode=True, sort_keys=False)
        if original != new:
            backup = p.with_name(p.name + ".pre_a1.bak")
            if not backup.exists():
                backup.write_text(original, encoding="utf-8")
            p.write_text(new, encoding="utf-8")

    # Remove old heavy outputs from the cleaned deliverable notebook.
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    bootstrap_cell = {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": bootstrap.splitlines(keepends=True)
    }
    nb["cells"].insert(CELL_MAP[8], bootstrap_cell)
    NEW_NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+"\n",encoding="utf-8")

    test_path = ROOT / "tests" / "test_a1_configuration.py"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    if not test_path.exists():
        test_path.write_text('''from pathlib import Path

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
    path = ROOT / "advanced-nlp-config-driven.ipynb"
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    code = "\\n".join("".join(c.get("source", [])) for c in cells)
    assert "from src.metrics import retrieval_metrics" in code
    assert "from src.retrieval import exact_cosine_search" in code
    assert "a1_prompt(DENSE_CFG)" in code
''',encoding="utf-8")

    print("A1 configuration finalized without running GPU models.")
    print("New notebook:", NEW_NOTEBOOK)
    print("Prompts:", *(f"prompts/{s}.txt" for s in SYSTEMS))
    print("Configs updated with backups (*.pre_a1.bak)")
    print("Code cells syntax-checked:",len(edited))
    print("Config unit tests:",test_path)
    print("Next: python -m pytest tests/ -v")
    print("Then: python src/prepare_phase1.py (refreshes run_config.yaml)")


if __name__ == "__main__":
    main()
