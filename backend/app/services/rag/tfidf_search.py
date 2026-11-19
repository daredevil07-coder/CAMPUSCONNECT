import logging
import math
import re
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower()) if len(w) > 1]


class TfidfSearchEngine:
    """Lightweight, pure-Python TF-IDF Search Engine without heavy scikit-learn dependency."""
    def __init__(self):
        self.doc_ids: List[str] = []
        self.doc_tf: List[Dict[str, float]] = []
        self.idf: Dict[str, float] = {}
        self.doc_norms: List[float] = []

    def fit_corpus(self, docs: List[Dict[str, Any]]) -> None:
        """Fit TF-IDF statistics across approved knowledge records."""
        if not docs:
            self.doc_ids = []
            self.doc_tf = []
            self.idf = {}
            self.doc_norms = []
            return

        self.doc_ids = []
        self.doc_tf = []
        doc_freqs: Dict[str, int] = {}
        total_docs = len(docs)

        for d in docs:
            parts = [
                d.get("canonical_question", ""),
                " ".join(d.get("alternative_questions", [])),
                " ".join(d.get("keywords", [])),
                d.get("answer", ""),
                d.get("content", ""),
            ]
            full_text = " ".join(filter(None, parts))
            tokens = _tokenize(full_text)
            
            tf_counts: Dict[str, int] = {}
            for t in tokens:
                tf_counts[t] = tf_counts.get(t, 0) + 1

            doc_len = len(tokens) or 1
            # Sublinear tf: 1 + log(count)
            tf_dict = {t: 1.0 + math.log(cnt) for t, cnt in tf_counts.items()}
            self.doc_tf.append(tf_dict)
            self.doc_ids.append(d["id"])

            for t in set(tokens):
                doc_freqs[t] = doc_freqs.get(t, 0) + 1

        # Compute IDF
        self.idf = {}
        for term, df in doc_freqs.items():
            self.idf[term] = math.log((1.0 + total_docs) / (1.0 + df)) + 1.0

        # Precompute document vector norms
        self.doc_norms = []
        for tf_dict in self.doc_tf:
            norm_sq = sum((tf_val * self.idf.get(t, 1.0)) ** 2 for t, tf_val in tf_dict.items())
            self.doc_norms.append(math.sqrt(norm_sq) if norm_sq > 0 else 1.0)

    def search(self, query: str, top_k: int = 10) -> Dict[str, float]:
        """Returns a mapping of doc_id -> tfidf_similarity_score in [0, 1]."""
        if not self.doc_ids or not self.idf:
            return {}

        query_tokens = _tokenize(query)
        if not query_tokens:
            return {}

        q_tf: Dict[str, int] = {}
        for t in query_tokens:
            q_tf[t] = q_tf.get(t, 0) + 1

        # Query vector: tf * idf
        q_vec = {}
        q_norm_sq = 0.0
        for t, cnt in q_tf.items():
            if t in self.idf:
                weight = (1.0 + math.log(cnt)) * self.idf[t]
                q_vec[t] = weight
                q_norm_sq += weight * weight

        q_norm = math.sqrt(q_norm_sq)
        if q_norm == 0:
            return {}

        results = {}
        for idx, doc_id in enumerate(self.doc_ids):
            tf_dict = self.doc_tf[idx]
            dot_product = 0.0
            for t, q_weight in q_vec.items():
                if t in tf_dict:
                    d_weight = tf_dict[t] * self.idf[t]
                    dot_product += q_weight * d_weight

            sim = dot_product / (q_norm * self.doc_norms[idx])
            results[doc_id] = float(max(0.0, min(1.0, sim)))

        return results
