import json
import math
import re
import numpy as np
from collections import Counter, defaultdict

WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)

def bm25_tokens(text):
    return WORD.findall(str(text).casefold())

def make_units(items, mode):
    assert mode in ('passage', 'title', 'sentence')
    units = []
    for p in items:
        doc = p['doc_id']
        if mode == 'passage':
            units.append((doc, p['text']))
        elif mode == 'title':
            units.append((doc, p['title'] + '\n' + p['text']))
        else:
            units.extend((doc, s) for s in p['sentences'] if s.strip())
    return units

class BM25:
    def __init__(self, units, k1=1.5, b=.75):
        self.units, self.k1, self.b = list(units), float(k1), float(b)
        self.lengths = []
        self.postings = defaultdict(list)
        for i, (_doc_id, text) in enumerate(self.units):
            tf = Counter(bm25_tokens(text))
            self.lengths.append(sum(tf.values()))
            for term, freq in tf.items():
                self.postings[term].append((i, freq))
        self.N = len(self.units)
        self.avgdl = sum(self.lengths) / self.N if self.N else 0.0
        self.idf = {term: math.log1p((self.N - len(postings) + 0.5) /
                                     (len(postings) + 0.5))
                    for term, postings in self.postings.items()}

    def search(self, query, k=10, exclude_ids=()):
        if k <= 0:
            return []
        excluded = set(exclude_ids)
        unit_scores = defaultdict(float)
        for term in set(bm25_tokens(query)):
            if term not in self.postings:  # OOV: contributes zero
                continue
            idf = self.idf[term]
            for i, freq in self.postings[term]:
                doc_id = self.units[i][0]
                if doc_id in excluded:
                    continue  # document mask BEFORE ranking
                normalization = self.k1 * (1-self.b + self.b*self.lengths[i]/(self.avgdl or 1))
                unit_scores[i] += idf * freq * (self.k1+1) / (freq + normalization)
        # Sentence max-pooling also works in passage/title mode (one unit per doc).
        document_scores = {}
        for i, value in unit_scores.items():
            doc_id = self.units[i][0]
            document_scores[doc_id] = max(value, document_scores.get(doc_id, 0.0))
        return [doc for doc, score in sorted(document_scores.items(),
                                          key=lambda x: (-x[1], x[0]))[:k]]


def rrf_fusion(ranked_lists, k=10, rank_constant=60):
    scores = defaultdict(float)
    for ranked in ranked_lists:
        seen = set()
        for rank, doc_id in enumerate(ranked, start=1):
            if doc_id not in seen:
                scores[doc_id] += 1.0 / (rank_constant + rank)
                seen.add(doc_id)
    return sorted(scores, key=lambda d: (-scores[d], d))[:k]


def parse_grounded_output(raw, allowed_ids):
    fallback = {'answer':'', 'cited':[], 'abstained':True}
    clean = re.sub(r'<think>.*?</think>', '', str(raw), flags=re.S | re.I)
    parser = json.JSONDecoder()
    allowed = set(allowed_ids)
    for found in re.finditer(r'\{', clean):
        try:
            obj, _ = parser.raw_decode(clean[found.start():])
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        answer, cited, abstained = obj.get('answer'), obj.get('cited'), obj.get('abstained')
        if not (isinstance(answer, str) and isinstance(cited, list)
                and isinstance(abstained, bool)
                and all(isinstance(x, str) for x in cited)):
            continue
        if abstained or not answer.strip():
            return fallback
        if not cited or any(x not in allowed for x in cited):
            return fallback
        return {'answer': answer.strip(), 'cited': list(dict.fromkeys(cited)),
                'abstained': False}
    return fallback



def passage_block(item):
    """Format one complete, citeable passage block (exact notebook format)."""
    return '[doc_id: {doc}]\nTitle: {title}\nText: {text}'.format(
        doc=item['doc_id'], title=item['title'], text=item['text'])


def pack_context(doc_ids, tokenizer, budget=1800, *, corpus_by_id):
    """Fit whole passage blocks, counting labels and separators using the LLM tokenizer."""
    blocks, selected = [], []
    for doc_id in dict.fromkeys(doc_ids):
        block = passage_block(corpus_by_id[doc_id])
        proposal = '\n\n'.join([*blocks, block])
        if len(tokenizer.encode(proposal, add_special_tokens=False)) <= budget:
            blocks.append(block)
            selected.append(doc_id)
    return '\n\n'.join(blocks), selected


def cosine_rank(query_vector, top_k, *, doc_vectors, doc_ids, excluded=()):
    """Exact cosine top-k for L2-normalized vectors, masking before selection."""
    sims = doc_vectors @ query_vector
    forbidden = set(excluded)
    if forbidden:
        sims[np.isin(doc_ids, list(forbidden))] = -np.inf
    order = np.lexsort((doc_ids, -sims))
    return [str(doc_ids[i]) for i in order[:top_k] if np.isfinite(sims[i])]
