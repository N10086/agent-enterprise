"""把用户上传的各种文档抽成纯文本。

**只用标准库 + pypdf**，不引入一堆解析库：
  - `.md/.markdown/.txt/.rst/.csv/.json`：本来就是文本，直接解码
  - `.docx` / `.pptx`：本质是 zip + XML，用 zipfile + ElementTree 取文字即可
  - `.pdf`：唯一需要第三方库的格式，用已有的 pypdf 抽文本

老格式 `.doc` / `.ppt`（OLE 二进制）不在这里支持——解析它们要额外依赖，
而且用户在 Office 里另存为 `.docx` / `.pptx` 只有一步。
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

#: MIME / 后缀 → 解析器 的映射（界面上的 accept 也按这份清单来）
TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".csv", ".json", ".log"}
ZIP_XML_SUFFIXES = {".docx", ".pptx"}
PDF_SUFFIXES = {".pdf"}

SUPPORTED_SUFFIXES = TEXT_SUFFIXES | ZIP_XML_SUFFIXES | PDF_SUFFIXES

#: 给界面展示用的说明
FORMAT_HINT = "支持 .pdf / .docx / .pptx / .md / .txt / .rst / .csv / .json"

MAX_FILE_BYTES = 20 * 1024 * 1024
#: 一次导入的总大小上限（上传接口按 base64 传输，留出余量）
MAX_BATCH_BYTES = 40 * 1024 * 1024


class DocumentError(Exception):
    """解析失败。消息直接展示给用户，所以要写人话。"""


def _decode(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _zip_text(data: bytes, members: list[str]) -> str:
    """从 docx/pptx 里按给定成员顺序取文字。"""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise DocumentError(f"文件不是有效的 Office 文档（zip 结构损坏）：{exc}") from exc

    blocks: list[str] = []
    for member in members:
        try:
            raw = archive.read(member)
        except KeyError:
            continue
        blocks.append(raw.decode("utf-8", errors="replace"))
    if not blocks:
        raise DocumentError("文档里没有找到正文（可能只包含图片或受保护）")
    return "\n".join(blocks)


def _xml_paragraphs(xml_text: str, text_tag_suffix: str, block_tag_suffix: str) -> list[str]:
    """按块（段落 / 幻灯片）抽文字，块之间保留空行，段落内的行内片段直接拼接。"""
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as exc:
        raise DocumentError(f"XML 解析失败：{exc}") from exc

    paragraphs: list[str] = []
    for block in root.iter():
        if not block.tag.endswith(block_tag_suffix):
            continue
        parts = [
            (node.text or "")
            for node in block.iter()
            if node.tag.endswith(text_tag_suffix) and (node.text or "").strip()
        ]
        # <w:br/> 之类的换行会被丢掉，但长文里影响可忽略
        line = "".join(parts).strip()
        if line:
            paragraphs.append(line)
    return paragraphs


def _pptx_text(data: bytes) -> str:
    """按幻灯片顺序取文字（slide1、slide2… 数字排序，而不是字典序）。"""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise DocumentError(f"文件不是有效的 PPTX（zip 结构损坏）：{exc}") from exc

    names = [
        name
        for name in archive.namelist()
        if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
    ]
    names.sort(key=lambda name: int(re.search(r"(\d+)", name).group(1)))
    if not names:
        raise DocumentError("PPTX 里没有找到幻灯片内容")

    chunks: list[str] = []
    for index, name in enumerate(names, start=1):
        xml_text = archive.read(name).decode("utf-8", errors="replace")
        lines = _xml_paragraphs(xml_text, "}t", "}sp")
        if lines:
            chunks.append(f"【第 {index} 页】\n" + "\n".join(lines))
    return "\n\n".join(chunks)


def _docx_text(data: bytes) -> str:
    xml_text = _zip_text(data, ["word/document.xml"])
    lines = _xml_paragraphs(xml_text, "}t", "}p")
    return "\n\n".join(lines)


def _pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 环境缺库时的可读提示
        raise DocumentError("服务端缺少 pypdf，无法解析 PDF：pip install pypdf") from exc

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise DocumentError(f"PDF 打开失败：{type(exc).__name__}: {exc}") from exc

    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            pages.append(f"【第 {index} 页】解析失败：{type(exc).__name__}")
            continue
        text = text.strip()
        if text:
            pages.append(f"【第 {index} 页】\n{text}")
    if not pages:
        raise DocumentError("PDF 里没有可提取的文字（可能是扫描件，需要 OCR）")
    return "\n\n".join(pages)


def extract_text(filename: str, data: bytes) -> str:
    """按后缀把文件抽成纯文本。失败抛 ``DocumentError``（消息可直出给用户）。"""
    if not data:
        raise DocumentError("文件是空的")
    if len(data) > MAX_FILE_BYTES:
        raise DocumentError(f"文件过大（{len(data) / 1048576:.1f}MB），单文件上限 20MB")

    suffix = Path(filename).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        text = _decode(data)
    elif suffix == ".docx":
        text = _docx_text(data)
    elif suffix == ".pptx":
        text = _pptx_text(data)
    elif suffix == ".pdf":
        text = _pdf_text(data)
    elif suffix in {".doc", ".ppt"}:
        raise DocumentError(
            f"暂不支持老格式 {suffix}：请在 Office 里另存为 {suffix}x 后再导入"
        )
    else:
        raise DocumentError(f"不支持的格式 {suffix or '(无后缀)'}。{FORMAT_HINT}")

    text = normalize(text)
    if len(text.strip()) < 20:
        raise DocumentError("解析出的文字太少，无法作为知识库内容")
    return text


def normalize(text: str) -> str:
    """统一换行、压掉多余空行——切分和嵌入都依赖段落结构稳定。"""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _cli() -> int:
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="把文档抽成纯文本（调试用）")
    parser.add_argument("path")
    args = parser.parse_args()
    target = Path(args.path)
    print(extract_text(target.name, target.read_bytes())[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = [
    "DocumentError",
    "FORMAT_HINT",
    "SUPPORTED_SUFFIXES",
    "extract_text",
    "normalize",
]
