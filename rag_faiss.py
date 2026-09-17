"""FAISS 本地向量库：嵌入、建库、持久化、检索。

**为什么用 FAISS 而不是 Chroma**：全本地、零服务、单文件索引，
检索是纯 numpy 级的矩阵运算（384 维、数百到数万块都是毫秒级），
适合做"检索质量"这类需要反复回放对比的实验。

索引用 `IndexFlatIP` + L2 归一化向量 ⇒ 内积等价于余弦相似度；
向量库规模在十万以内时，暴力检索比 IVF/HNSW 更准也更快。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import faiss
import numpy as np

#: 与 sentence-transformers/all-MiniLM-L6-v2 的输出维度一致
DEFAULT_DIM = 384


class Embedder:
    """本地嵌入模型（首次使用后走本地缓存，可离线运行）。"""

    def __init__(self, model_name: str, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        self.calls = 0
        self.texts_embedded = 0
        self.seconds = 0.0

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入并做 L2 归一化（归一化后内积即余弦相似度）。"""
        if not texts:
            return []
        start = time.time()
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.seconds += time.time() - start
        self.calls += 1
        self.texts_embedded += len(texts)
        return vectors.astype("float32").tolist()


class FaissStore:
    """FAISS 索引 + 块元数据。"""

    def __init__(self, index: faiss.Index, chunks: list[dict], dim: int = DEFAULT_DIM):
        self.index = index
        self.chunks = chunks
        self.dim = dim

    def __len__(self) -> int:
        return len(self.chunks)

    # ---- 建库 ----------------------------------------------------------
    @classmethod
    def build(cls, chunks: list[dict], embedder: Embedder) -> "FaissStore":
        """用块列表建索引。chunks 每项需含 content 与 metadata。"""
        vectors = np.array(embedder.embed([c["content"] for c in chunks]), dtype="float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, chunks, vectors.shape[1])

    @classmethod
    def load(cls, directory: str | Path) -> "FaissStore":
        directory = Path(directory)
        index = faiss.read_index(str(directory / "index.faiss"))
        chunks = [
            json.loads(line)
            for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(index, chunks, index.d)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        return directory

    # ---- 检索 ----------------------------------------------------------
    def search_one(self, query: str, k: int, embedder: Embedder) -> list[tuple[float, dict]]:
        """单条查询，返回 [(相似度, 块)]，按相似度降序。"""
        vector = np.array(embedder.embed([query]), dtype="float32")
        scores, indices = self.index.search(vector, min(k, len(self.chunks)))
        return [
            (float(score), self.chunks[int(idx)])
            for score, idx in zip(scores[0], indices[0])
            if int(idx) >= 0
        ]

    def search_many(self, queries: list[str], k: int, embedder: Embedder) -> list[list[tuple[float, dict]]]:
        """多条查询各自的检索结果（不合并）。"""
        return [self.search_one(query, k, embedder) for query in queries]


def rrf_merge(result_lists: list[list[tuple[float, dict]]], k: int, rrf_k: int = 60) -> list[tuple[float, dict]]:
    """用 RRF（倒数排名融合）合并多路检索结果。

    只依赖"在各自排名里第几"，不依赖相似度绝对值——
    原始查询与 HyDE 假设答案的分数分布不可比，直接加权是错的。
    """
    scores: dict[str, float] = {}
    best: dict[str, tuple[float, dict]] = {}
    for results in result_lists:
        for rank, (score, chunk) in enumerate(results, start=1):
            key = chunk["metadata"].get("chunk_id") or chunk["content"][:80]
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            if key not in best or score > best[key][0]:
                best[key] = (score, chunk)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    merged = []
    for key, rrf_score in ordered:
        score, chunk = best[key]
        item = dict(chunk)
        item["rrf_score"] = round(rrf_score, 6)
        item["score"] = round(score, 4)
        merged.append((score, item))
    return merged
