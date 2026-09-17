"""多模型接入层：provider 注册表 + "本次请求用哪个模型"。

项目里所有 LLM 调用（图里的 researcher / analyst / grade_documents，以及 RAG 链路
内部的 multi-query / HyDE）都从这里取模型，所以**换模型只需要换一个 contextvar**，
不用改任何节点代码：`with use_llm("qwen", "qwen-plus", api_key=...):` 一次生效。

各家的接口都兼容 OpenAI 协议，差别只在 `base_url` / `api_key` / `model` 三件事，
因此统一用 `ChatOpenAI` 指向不同端点，而不是给每家写一套适配。

密钥优先级：请求里带的 key → 该 provider 的环境变量 → 项目默认的 `OPENAI_API_KEY`。
"""
from __future__ import annotations

import contextlib
import contextvars
import os
from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from .config import get_settings

settings = get_settings()


@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    base_url: str
    env_key: str
    models: tuple[str, ...]
    note: str = ""


#: 需要新增一家时只在这里加一行：任何 OpenAI 兼容端点都能接
PROVIDERS: tuple[Provider, ...] = (
    Provider(
        key="deepseek",
        label="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        env_key="DEEPSEEK_API_KEY",
        models=("deepseek-chat", "deepseek-reasoner"),
    ),
    Provider(
        key="qwen",
        label="通义千问 Qwen",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        env_key="DASHSCOPE_API_KEY",
        models=("qwen-plus", "qwen-max", "qwen-turbo", "qwen-long"),
    ),
    Provider(
        key="glm",
        label="智谱 GLM",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        env_key="ZHIPUAI_API_KEY",
        models=("glm-4-plus", "glm-4-air", "glm-4-flash"),
    ),
    Provider(
        key="kimi",
        label="Kimi（Moonshot）",
        base_url="https://api.moonshot.cn/v1",
        env_key="MOONSHOT_API_KEY",
        models=("moonshot-v1-8k", "moonshot-v1-32k", "kimi-k2-0711-preview"),
    ),
    Provider(
        key="openai",
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        # 刻意不用 OPENAI_API_KEY：.env 里那个变量通常指向别家兼容端点，
        # 借用它会把 A 家的 key 发到 OpenAI。真正配了 OpenAI 时，
        # 下面的 _env_key 会按"端点同源"规则自动复用 .env 的 key。
        env_key="OPENAI_DIRECT_API_KEY",
        models=("gpt-4o-mini", "gpt-4o", "gpt-4.1"),
    ),
    Provider(
        key="ollama",
        label="本地 Ollama",
        base_url="http://127.0.0.1:11434/v1",
        env_key="OLLAMA_API_KEY",
        models=("qwen2.5:7b", "llama3.1:8b"),
        note="本地服务，key 随便填（例如 ollama）；需先 ollama serve",
    ),
    Provider(
        key="custom",
        label="自定义（OpenAI 兼容）",
        base_url="",
        env_key="CUSTOM_API_KEY",
        models=(),
        note="在界面上填 Base URL 与模型名即可",
    ),
)

PROVIDER_MAP = {item.key: item for item in PROVIDERS}


def get_provider(key: str | None) -> Provider:
    return PROVIDER_MAP.get((key or "deepseek").strip().lower(), PROVIDER_MAP["deepseek"])


def _same_host(left: str, right: str) -> bool:
    from urllib.parse import urlparse

    try:
        return bool(urlparse(left).netloc) and urlparse(left).netloc == urlparse(right).netloc
    except ValueError:
        return False


def _env_key(spec: Provider) -> str:
    """该 provider 可用的环境变量 key。

    除了自己的变量，还认"`.env` 配的就是同一家端点"的情况：
    `.env` 里习惯写 `OPENAI_API_KEY` + `OPENAI_API_BASE`，如果它指向的正是
    api.deepseek.com，那么选 DeepSeek 时直接复用这把 key，不用再填一次。
    不同端点则不借用，避免把 A 家的 key 发到 B 家去。
    """
    key = os.getenv(spec.env_key, "").strip()
    if key:
        return key
    if settings.api_key and spec.base_url and _same_host(spec.base_url, settings.base_url):
        return settings.api_key.strip()
    return ""


def resolve_credentials(
    provider_key: str | None, api_key: str | None = None, base_url: str | None = None
) -> tuple[str, str]:
    """返回 ``(api_key, base_url)``；都没有则抛错，让调用方给出可读提示。"""
    provider = get_provider(provider_key)
    key = (api_key or "").strip() or _env_key(provider)
    url = (base_url or "").strip() or provider.base_url or ""
    return key, url


def build_llm(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
) -> ChatOpenAI:
    """构造某一家、某个型号的 LLM。"""
    spec = get_provider(provider)
    key, url = resolve_credentials(spec.key, api_key, base_url)
    if not key:
        raise ValueError(
            f"未配置 {spec.label} 的 API Key：请在界面上填入，或设置环境变量 {spec.env_key}"
        )
    chosen = (model or "").strip() or (spec.models[0] if spec.models else settings.model)
    return ChatOpenAI(
        model=chosen,
        api_key=key,
        base_url=url or None,
        temperature=temperature,
    )


def available_providers() -> list[dict]:
    """给界面用的 provider 清单，并标明"服务端是否已有该家的 key"。"""
    result = []
    for spec in PROVIDERS:
        result.append(
            {
                "key": spec.key,
                "label": spec.label,
                "base_url": spec.base_url,
                "models": list(spec.models),
                "has_key": bool(_env_key(spec)),
                "env_key": spec.env_key,
                "note": spec.note,
            }
        )
    return result


# ---------------------------------------------------------------- 当前生效的模型

_ACTIVE: contextvars.ContextVar = contextvars.ContextVar("active_llm", default=None)
_DEFAULT: ChatOpenAI | None = None


def default_llm() -> ChatOpenAI:
    """没有显式指定时的默认模型（沿用 .env 配置，供 CLI / 评测使用）。"""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ChatOpenAI(
            model=settings.model,
            api_key=settings.api_key,
            base_url=settings.base_url or None,
            temperature=settings.temperature,
        )
    return _DEFAULT


def get_active_llm() -> ChatOpenAI:
    """当前上下文生效的模型；没设置就回落到默认。"""
    return _ACTIVE.get() or default_llm()


@contextlib.contextmanager
def use_llm(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
):
    """在这个上下文内，整张图（含 RAG 链路内部调用）都用指定的模型。

    用 contextvar 而不是全局变量：本地服务是多线程的，两个浏览器标签同时用不同
    模型时不能互相串味。
    """
    if not provider and not model and not api_key and not base_url:
        yield get_active_llm()
        return
    llm = build_llm(provider, model, api_key, base_url, temperature)
    token = _ACTIVE.set(llm)
    try:
        yield llm
    finally:
        _ACTIVE.reset(token)


class ActiveLLM:
    """模块级 `llm` 的替身：每次调用都转发给当前生效的模型。

    节点里写的是 `llm.invoke(...)` / `llm.bind_tools(...)`，换成这个替身之后
    一行都不用改，但模型可以在请求级别切换。
    """

    def __getattr__(self, item):
        return getattr(get_active_llm(), item)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<ActiveLLM {get_active_llm().model_name}>"


__all__ = [
    "ActiveLLM",
    "PROVIDERS",
    "available_providers",
    "build_llm",
    "default_llm",
    "get_active_llm",
    "get_provider",
    "resolve_credentials",
    "use_llm",
]
