"""Token 统计：给旧 Agent 管线补上效率指标。

旧管线的结果里只有准确率和工具轨迹，没有 token / 上下文消耗，
导致"新管线省不省、贵不贵"这件事无法回答。这里用 LangChain 回调
挂在图的 LLM 调用上，把每次调用的输入/输出 token 累计起来。

不同 provider 把用量放在不同位置（``usage_metadata`` 或 ``llm_output``），
两种都读，读不到就记 0，不会因为缺少用量而中断评测。
"""
from __future__ import annotations

from langchain_core.callbacks import BaseCallbackHandler


class TokenCounter(BaseCallbackHandler):
    """累计一次评测（或一道题）内所有 LLM 调用的 token。"""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        #: 每次调用的用量明细，按发生顺序：做成本归因（改写 / HyDE / 作答各占多少）时必需
        self.records: list[dict] = []

    # ---- 回调入口 ------------------------------------------------------
    def on_llm_end(self, response, **kwargs) -> None:  # noqa: D102
        self.calls += 1
        usage = self._usage_from_response(response)
        self.input_tokens += usage[0]
        self.output_tokens += usage[1]
        self.records.append(
            {
                "call": self.calls,
                "input_tokens": usage[0],
                "output_tokens": usage[1],
                "total_tokens": usage[0] + usage[1],
            }
        )

    # ---- 解析用量 ------------------------------------------------------
    @staticmethod
    def _usage_from_response(response) -> tuple[int, int]:
        # 1) 消息级 usage_metadata（OpenAI 兼容接口通常带这个）
        for generations in getattr(response, "generations", []) or []:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None)
                if usage:
                    return (
                        int(usage.get("input_tokens", 0) or 0),
                        int(usage.get("output_tokens", 0) or 0),
                    )
        # 2) 响应级 llm_output.token_usage
        llm_output = getattr(response, "llm_output", None) or {}
        token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if token_usage:
            return (
                int(token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0)) or 0),
                int(
                    token_usage.get("completion_tokens", token_usage.get("output_tokens", 0)) or 0
                ),
            )
        return 0, 0

    # ---- 读取 ----------------------------------------------------------
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def snapshot(self) -> dict:
        return {
            "llm_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "calls": list(self.records),
        }

    def reset(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.records = []
