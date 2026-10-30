import os
import shutil
import tempfile
from pathlib import Path

import faiss
import numpy as np

from .ingest import load_documents
from .io import ROOT, digest, now, read_json, write_json

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model=None):
        from sentence_transformers import SentenceTransformer
        self.name = model or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
        self.model = SentenceTransformer(self.name, device="cpu")
        self.tokenizer = self.model.tokenizer
        self.window = self.model.max_seq_length - self.tokenizer.num_special_tokens_to_add(False)
        if self.window < 8:
            raise ValueError("Embedding model sequence window is too short")

    def tokens(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False, truncation=False)

    def encode(self, texts):
        # MiniLM supports 256 tokens. Pool multiple nontruncated model windows so
        # 512/1024-token experimental chunks actually contribute their full text.
        windows, spans = [], []
        for text in texts:
            encoded = self.tokenizer(text, add_special_tokens=False, truncation=False,
                                     return_offsets_mapping=True)
            ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
            if not ids:
                raise ValueError("Cannot embed empty text")
            start = len(windows)
            begin = 0
            while begin < len(ids):
                end = min(begin + self.window, len(ids))
                piece = text[offsets[begin][0]:offsets[end-1][1]]
                # A subwindow can start inside a WordPiece word. Retokenizing that
                # substring may expand it; shrink until it fits, without losing
                # the tail or silently letting SentenceTransformer truncate it.
                while len(self.tokens(piece)) > self.window and end > begin + 1:
                    end -= 1
                    piece = text[offsets[begin][0]:offsets[end-1][1]]
                if len(self.tokens(piece)) > self.window:
                    raise ValueError("Single source token cannot fit the embedding model")
                windows.append(piece)
                begin = end
            spans.append((start, len(windows)))
        if not spans:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype="float32")
        vectors = self.model.encode(windows, batch_size=16, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False)
        # Equal-weight normalized subwindow means; documented experimental control.
        pooled = np.stack([vectors[a:b].mean(axis=0) for a, b in spans]).astype("float32")
        faiss.normalize_L2(pooled)
        return pooled


def chunk_documents(documents, tokenizer, size=512, overlap_ratio=0.125):
    if size <= 0 or not 0 <= overlap_ratio < 1:
        raise ValueError("Positive size and 0 <= overlap_ratio < 1 required")
    overlap = int(size * overlap_ratio)
    step = size - overlap
    chunks = []
    for doc in documents:
        encoded = tokenizer(doc["text"], add_special_tokens=False, truncation=False,
                            return_offsets_mapping=True)
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        for start in range(0, len(ids), step):
            end = min(start + size, len(ids))
            a, b = offsets[start][0], offsets[end-1][1]
            text = doc["text"][a:b]
            chunks.append({"chunk_id": f"{doc['id']}:{start}-{end}", "document_id": doc["id"],
                           "title": doc["title"], "url": doc["url"], "topic": doc["topic"],
                           "token_start": start, "token_end": end, "char_start": a, "char_end": b,
                           "text": text})
            if end == len(ids):
                break
    return chunks


def corpus_fingerprint(documents):
    return digest("\n".join(sorted(d["id"] + ":" + digest(d["text"].encode()) for d in documents)).encode())


def build_index(size=512, destination=None, documents=None, embedder=None):
    documents = documents if documents is not None else load_documents()
    embedder = embedder or Embedder()
    chunks = chunk_documents(documents, embedder.tokenizer, size)
    if not chunks:
        raise ValueError("No chunks to index")
    vectors = embedder.encode([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    destination = Path(destination or ROOT / f"artifacts/index-{size}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".index-", dir=destination.parent))
    try:
        faiss.write_index(index, str(temp / "vectors.faiss"))
        write_json(temp / "chunks.json", chunks)
        if isinstance(embedder, TfidfEmbedder):
            write_json(temp / "tfidf.json", embedder.state())
        write_json(temp / "metadata.json", {
            "created_at": now(), "model": embedder.name,
            "model_commit": getattr(embedder.model[0].auto_model.config, "_commit_hash", None) if isinstance(embedder, Embedder) else None,
            "chunk_tokens": size, "overlap_ratio": 0.125,
            "pooling": "equal mean of normalized nontruncated model-window vectors, then L2 normalize" if isinstance(embedder, Embedder) else "TF-IDF L2 normalized, vocabulary fit on full documents",
            "embedding_window_tokens": embedder.window, "dimension": vectors.shape[1],
            "documents": len(documents), "chunks": len(chunks),
            "corpus_sha256": corpus_fingerprint(documents),
        })
        if destination.exists():
            shutil.rmtree(destination)
        temp.rename(destination)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return destination


class Retriever:
    def __init__(self, directory, embedder=None):
        directory = Path(directory)
        self.metadata = read_json(directory / "metadata.json")
        self.embedder = embedder or (TfidfEmbedder(state=read_json(directory / "tfidf.json"))
                                    if self.metadata["model"] == TfidfEmbedder.name else Embedder(self.metadata["model"]))
        if self.embedder.name != self.metadata["model"]:
            raise ValueError("Query/index embedding models differ; rebuild index")
        self.index = faiss.read_index(str(directory / "vectors.faiss"))
        self.chunks = read_json(directory / "chunks.json")
        if self.index.ntotal != len(self.chunks):
            raise ValueError("Index/chunk count mismatch")

    def search(self, query, k=5):
        if not query.strip() or k < 1:
            raise ValueError("Nonempty query and positive k required")
        vector = self.embedder.encode([query])
        if not np.any(vector):
            # TF-IDF has no cross-language/OOV semantics; never return arbitrary ties.
            return []
        scores, ids = self.index.search(vector, min(k, len(self.chunks)))
        return [{**self.chunks[int(i)], "score": float(s)} for i, s in zip(ids[0], scores[0]) if i >= 0]


class TfidfEmbedder:
    """Explicit diagnostic baseline. Not a substitute for MiniLM validation."""
    name = "tfidf-word-1-2-baseline"

    def __init__(self, documents=None, state=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
        self.window = None
        self.vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), sublinear_tf=True)
        if state:
            self.vectorizer.set_params(vocabulary=state["vocabulary"])
            self.vectorizer.fit(["placeholder"])
            self.vectorizer.idf_ = np.asarray(state["idf"], dtype="float64")
        else:
            self.vectorizer.fit([d["text"] for d in documents])

    def encode(self, texts):
        return self.vectorizer.transform(texts).toarray().astype("float32")

    def state(self):
        return {"vocabulary": {k: int(v) for k, v in self.vectorizer.vocabulary_.items()},
                "idf": self.vectorizer.idf_.tolist()}
