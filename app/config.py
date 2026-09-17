import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")
load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    """运行时配置。所有可调项都能用环境变量覆盖。"""

    # ---- 模型 ----
    model: str = os.getenv("MODEL_NAME", "deepseek-chat")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    base_url: str = os.getenv("OPENAI_API_BASE", os.getenv("OPENAI_BASE_URL", ""))
    temperature: float = 0.0

    # ---- 本地知识库（RAG） ----
    # 继续使用公开评测目录中的知识库，避免额外的 source_knowledge 目录
    # 语料目录：可入库的 md/txt/pdf 都放这里，由 build_kb.py 生成
    kb_corpus_dir: str = os.getenv("KB_CORPUS_DIR", str(BASE_DIR / "public" / "corpus"))
    # 向量库位置：保持与 public 目录同级，避免绕过公开评测数据
    kb_vector_store: str = os.getenv("KB_VECTOR_STORE", str(BASE_DIR / "public" / "vector_store"))
    # 语料是英文维基段落，用英文嵌入模型；换成中文语料时改回 BAAI/bge-small-zh-v1.5
    kb_embedding_model: str = os.getenv("KB_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # 向量库为空时是否允许自动入库；关掉可避免 Agent 运行途中触发耗时入库
    kb_auto_ingest: bool = _env_bool("KB_AUTO_INGEST", True)
    # 实测：本语料下 top-3 召回 96%、top-5 召回 100%，故默认取 5
    kb_top_k: int = _env_int("KB_TOP_K", 5)
    # 检索后端：auto（存在 FAISS 语义索引就用它，否则回退旧的 Chroma 库）
    kb_backend: str = os.getenv("KB_BACKEND", "auto")
    # 语义切分 + FAISS 索引位置（由 build_rag_kb.py 生成）
    kb_faiss_index: str = os.getenv(
        "KB_FAISS_INDEX", str(BASE_DIR / "public" / "faiss_kb_semantic")
    )
    # 检索质量检查（adaptive-RAG / CRAG 里的 grading 步）：rag_search 取回的片段
    # 若被判定与问题无关，就回到 supervisor 换查询重检一轮，而不是硬着头皮作答。
    # 关掉可省一次 LLM 调用（KB_GRADE_RETRIEVAL=0），代价是失去这层纠错。
    kb_grade_retrieval: bool = _env_bool("KB_GRADE_RETRIEVAL", True)
    # 质检的触发线：只有当这批片段的**最高相似度**低于它时，才值得花一次 LLM 调用
    # 去质检；相似度高的检索直接进作答，省掉这笔钱。
    # 注意：相似度在这里只决定"要不要质检"，**绝不**用来丢弃证据——
    # 质检由 LLM 做语义判断，判定不相关时采取的动作是"换查询重检"，不是直接拒答。
    # 设为 0 表示每次都质检。
    kb_grade_score: float = _env_float("KB_GRADE_SCORE", 0.55)

    # ---- MCP（Model Context Protocol）----
    # 外部 MCP server 配置（JSON），把别人的 MCP 工具接进本项目的 Agent 图：
    #   {"<名字>": {"command": "...", "args": [...], "env": {...}}}
    #   {"<名字>": {"url": "http://host:port/mcp"}}      # streamable-http
    # 留空表示不接外部工具，图的行为与没有 MCP 时完全一致。
    mcp_servers: str = os.getenv("MCP_SERVERS", "")

    # ---- 被复用的 RAG 实现所在项目 ----
    enterprise_root: str = os.getenv(
        "ENTERPRISE_ROOT", str(BASE_DIR.parent / "enterprise-research-agent")
    )


settings = Settings()


def get_settings() -> Settings:
    return settings
