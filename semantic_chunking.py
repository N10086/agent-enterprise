"""语义切分：按语义边界把文档切成 400~800 字符的块。

**为什么不用固定字数切**：固定窗口会把一个完整的意思拦腰截断——
"XX 出生于 A 地" 和 "他后来在 B 地任职" 被切到两块，检索时就只剩半句。

**做法**：先切成句子，用嵌入模型算相邻句子的余弦相似度；
相似度低的地方说明语义发生了转折，是天然切点。

切分规则（三条同时满足）：
  1. 块长目标 400~800 字符；
  2. 只在"语义低谷"处切：该处相似度低于本文档的候选阈值；
  3. 不足 400 字符不切；超过 800 字符必须切，切点取 800 之前语义最弱的那一处。

**已知例外**：语料里存在本身不足 400 字符的短段落（维基短条目），
这时保留为独立块并标记 `short=True`，而不是跨标题拼接——
跨标题拼接会让检索结果无法归因到某一个条目。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

#: 目标块长区间
MIN_CHARS = 400
MAX_CHARS = 800
#: 语义低谷的判定分位：相邻相似度低于该分位视为候选切点
LOW_SIM_PERCENTILE = 0.35
#: 相似度低于此绝对值时无论分位都视为强边界
STRONG_BOUNDARY = 0.35

_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|\n{2,}")


@dataclass
class Chunk:
    """一个待入库的文本块。"""

    content: str
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return str(self.metadata.get("chunk_id", ""))

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", ""))


def split_sentences(text: str) -> list[str]:
    """按句末标点与空行切句，并丢掉空片段。"""
    return [part.strip() for part in _SENTENCE_RE.split(text or "") if part and part.strip()]


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * ratio)))
    return ordered[index]


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    """单句超长时按逗号/空格兜底硬切。"""
    pieces, current = [], ""
    for token in re.split(r"(?<=[,;，；])\s*", sentence):
        if current and len(current) + len(token) > max_chars:
            pieces.append(current)
            current = token
        else:
            current = token if not current else f"{current}{token}"
    if current:
        while len(current) > max_chars:
            pieces.append(current[:max_chars])
            current = current[max_chars:]
        if current:
            pieces.append(current)
    return pieces


def _best_cut(buffer: list[str], boundaries: list[float], start: int) -> int:
    """在 buffer 内部选一个切点下标（返回切在第几个句子之后）。

    优先取 800 字符以内相似度最低的边界；找不到就退回最长可行位置。
    """
    lengths, total = [], 0
    for sentence in buffer:
        total += len(sentence)
        lengths.append(total)

    candidates = [
        i
        for i in range(len(buffer) - 1)
        if lengths[i] >= MIN_CHARS and boundaries[start + i] <= 0
    ]
    if candidates:
        return min(candidates, key=lambda i: boundaries[start + i]) + 1

    for i in range(len(buffer) - 1):
        if lengths[i] >= MIN_CHARS:
            return i + 1
    return 1


def split_semantic(
    text: str,
    title: str,
    source: str,
    embed,
    min_chars: int = MIN_CHARS,
    max_chars: int = MAX_CHARS,
) -> list[Chunk]:
    """把一个段落的文本按语义切成 400~800 字符的块。

    Args:
        text: 段落正文
        title: 该段落的标题（维基条目名），写进元数据用于检索归因
        source: 来源标识
        embed: 批量嵌入函数，``embed(list[str]) -> list[list[float]]``
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    expanded: list[str] = []
    for sentence in sentences:
        expanded.extend(_hard_split(sentence, max_chars) if len(sentence) > max_chars else [sentence])

    # 整段足够短就不切（短条目原样保留，标记 short）
    body = " ".join(expanded)
    if len(body) <= max_chars:
        return [
            Chunk(
                content=body,
                metadata={"source": source, "title": title, "chunk_id": f"{title}::0",
                          "short": len(body) < min_chars},
            )
        ]

    vectors = embed(expanded) if len(expanded) > 1 else []
    similarities: list[float] = []
    for left, right in zip(vectors, vectors[1:]):
        similarities.append(sum(a * b for a, b in zip(left, right)))

    # 低于该分位（或绝对值很低）的边界算"语义低谷"，标 0 表示可切
    threshold = _percentile(similarities, LOW_SIM_PERCENTILE) if similarities else 0.0
    boundaries = [
        0 if (sim <= threshold or sim < STRONG_BOUNDARY) else 1 for sim in similarities
    ]

    chunks: list[Chunk] = []
    buffer: list[str] = []
    start = 0
    index = 0

    def flush(buffer_len: int) -> None:
        nonlocal buffer, index, start
        if not buffer:
            return
        body = " ".join(buffer).strip()
        # 拼接空格会让实际长度略大于按句子累计的长度，超限时按空格回退切分
        while len(body) > MAX_CHARS:
            cut = body.rfind(" ", 0, MAX_CHARS)
            cut = cut if cut > 0 else MAX_CHARS
            head, body = body[:cut].strip(), body[cut:].strip()
            if not head:
                break
            chunks.append(
                Chunk(
                    content=head,
                    metadata={
                        "source": source,
                        "title": title,
                        "chunk_id": f"{title}::{index}",
                        "short": len(head) < min_chars,
                    },
                )
            )
            index += 1
        if body:
            chunks.append(
                Chunk(
                    content=body,
                    metadata={
                        "source": source,
                        "title": title,
                        "chunk_id": f"{title}::{index}",
                        "short": len(body) < min_chars,
                    },
                )
            )
            index += 1
        start += buffer_len
        buffer = []

    length = 0
    for position, sentence in enumerate(expanded):
        if length and length + len(sentence) > max_chars:
            cut = _best_cut(buffer, boundaries, start)
            head = buffer[:cut]
            flush(cut)
            buffer = buffer[cut:] if cut < len(buffer) else []
            length = sum(len(s) for s in buffer)
        buffer.append(sentence)
        length += len(sentence)

        at_boundary = position < len(boundaries) and boundaries[position] == 0
        if at_boundary and length >= min_chars:
            flush(len(buffer))
            length = 0

    flush(len(buffer))
    return chunks


def _cut_point(text: str, min_chars: int, max_chars: int) -> int:
    """在 [min_chars, max_chars] 内找一个尽量自然的切点（优先句末，其次空格）。"""
    window = text[min_chars:max_chars]
    for pattern in (r"[.!?。！？]\s", r"[,;，；]\s", r"\s"):
        import re as _re

        matches = list(_re.finditer(pattern, window))
        if matches:
            return min_chars + matches[-1].end()
    return max_chars


def pack_chunks(
    chunks: list[Chunk], min_chars: int = MIN_CHARS, max_chars: int = MAX_CHARS
) -> list[Chunk]:
    """把过短的块与相邻块合并，尽量让每块落在 min~max 字符区间。

    语料里有不少不足 400 字符的短条目，若原样保留会让"块长 400~800"这条
    要求形同虚设。这里按原顺序合并，并为合并后的块保留全部来源标题
    （``titles``），所以检索结果仍然能归因到具体条目。

    当"短块 + 下一块"直接合并会超过 max_chars 时，不是硬留一个短块，
    而是**合并后按自然边界重新切开**，让两半都尽量落进区间。
    """
    packed: list[Chunk] = []
    buffer: list[Chunk] = []
    length = 0

    def emit(content: str, sources: list[Chunk], is_short: bool) -> None:
        if not content.strip():
            return
        titles: list[str] = []
        for item in sources:
            for one in item.metadata.get("titles") or [item.metadata.get("title", "")]:
                if one and one not in titles:
                    titles.append(one)
        metadata = dict(sources[0].metadata)
        metadata["title"] = titles[0] if titles else ""
        metadata["titles"] = titles
        metadata["merged_count"] = len(sources)
        metadata["short"] = is_short
        packed.append(Chunk(content=content.strip(), metadata=metadata))

    def flush() -> None:
        nonlocal buffer, length
        if not buffer:
            return
        emit(" ".join(item.content for item in buffer), buffer, length - 1 < min_chars)
        buffer, length = [], 0

    for chunk in chunks:
        if buffer and length + len(chunk.content) + 1 > max_chars:
            if length - 1 < min_chars:
                # 短块 + 下一块会超限：合并后重新切分，避免留下短块
                combined = " ".join(item.content for item in buffer) + " " + chunk.content
                cut = _cut_point(combined, min_chars, max_chars)
                emit(combined[:cut], buffer + [chunk], False)
                remainder = combined[cut:].strip()
                buffer = [Chunk(content=remainder, metadata=dict(chunk.metadata))] if remainder else []
                length = len(remainder) + 1
                if length - 1 >= min_chars:
                    flush()
                continue
            flush()
        buffer.append(chunk)
        length += len(chunk.content) + 1
        if length - 1 >= min_chars:
            flush()

    flush()
    for position, item in enumerate(packed):
        item.metadata["chunk_id"] = f"{item.metadata.get('title', '')}::{position}"
    return packed
