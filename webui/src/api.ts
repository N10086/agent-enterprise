/** 后端接口封装：普通 JSON 请求 + SSE 流式对话 + 浏览器本地保存的模型选择。 */
import type {
  Bootstrap,
  ChatRequest,
  Conversation,
  ConversationSummary,
  DirListing,
  KbState,
  ModelChoice,
  ProvidersPayload,
  StreamEvent,
  UploadResult,
  WorkspaceInfo,
} from "./types";

async function asJSON<T>(response: Response): Promise<T> {
  const text = await response.text();
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error(`响应不是 JSON（HTTP ${response.status}）：${text.slice(0, 200)}`);
  }
  if (!response.ok) {
    const message = (parsed as { error?: string })?.error ?? `HTTP ${response.status}`;
    throw new Error(message);
  }
  return parsed as T;
}

function post<T>(url: string, body?: unknown): Promise<T> {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  }).then((response) => asJSON<T>(response));
}

const CHOICE_KEY = "agent-ui-model";
const KEYS_KEY = "agent-ui-keys";

export const api = {
  bootstrap: () => fetch("/api/bootstrap").then((r) => asJSON<Bootstrap>(r)),
  providers: () => fetch("/api/providers").then((r) => asJSON<ProvidersPayload>(r)),

  createConversation: (title = "新对话") =>
    post<{ ok: boolean; conversation: Conversation; conversations: ConversationSummary[] }>(
      "/api/conversations",
      { title },
    ),
  getConversation: (id: string) =>
    fetch(`/api/conversations/${encodeURIComponent(id)}`).then((r) => asJSON<Conversation>(r)),
  deleteConversation: (id: string) =>
    post<{ ok: boolean; conversations: ConversationSummary[] }>(
      `/api/conversations/${encodeURIComponent(id)}/delete`,
    ),

  listDir: (path: string) => post<DirListing>("/api/fs/list", { path }),
  setFolder: (path: string) =>
    post<{ ok: boolean; workspace: WorkspaceInfo; kb: KbState }>("/api/workspace/folder", { path }),

  uploadKb: (files: { name: string; data: string }[]) =>
    post<UploadResult>("/api/kb/upload", { files }),
  excludeKbFile: (name: string) => post<UploadResult>("/api/kb/exclude", { name }),
  includeKbFile: (name: string) => post<UploadResult>("/api/kb/include", { name }),
  clearKb: () => post<UploadResult>("/api/kb/clear", {}),

  /** 把文件读成 base64；分块转换，避免大文件把调用栈撑爆。 */
  async readAsBase64(file: File): Promise<string> {
    const buffer = new Uint8Array(await file.arrayBuffer());
    const CHUNK = 0x8000;
    let binary = "";
    for (let i = 0; i < buffer.length; i += CHUNK) {
      binary += String.fromCharCode(...buffer.subarray(i, i + CHUNK));
    }
    return btoa(binary);
  },

  // ---- 模型选择（localStorage，两个窗口共用） ----
  loadChoice(): ModelChoice {
    try {
      const raw = JSON.parse(localStorage.getItem(CHOICE_KEY) ?? "{}") as Partial<ModelChoice>;
      const keys = JSON.parse(localStorage.getItem(KEYS_KEY) ?? "{}") as Record<string, string>;
      const provider = raw.provider ?? "deepseek";
      return {
        provider,
        model: raw.model ?? "",
        api_key: keys[provider] ?? null,
        base_url: raw.base_url ?? null,
      };
    } catch {
      return { provider: "deepseek", model: "", api_key: null, base_url: null };
    }
  },
  saveChoice(choice: ModelChoice): void {
    const keys = (() => {
      try {
        return JSON.parse(localStorage.getItem(KEYS_KEY) ?? "{}") as Record<string, string>;
      } catch {
        return {};
      }
    })();
    if (choice.api_key) keys[choice.provider] = choice.api_key;
    else delete keys[choice.provider];
    localStorage.setItem(KEYS_KEY, JSON.stringify(keys));
    localStorage.setItem(
      CHOICE_KEY,
      JSON.stringify({
        provider: choice.provider,
        model: choice.model,
        base_url: choice.base_url,
      }),
    );
  },
};

/** 跑一次对话，按 SSE 事件逐条回调。 */
export async function streamChat(
  payload: ChatRequest,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok || !response.body) {
    const text = await response.text().catch(() => "");
    throw new Error(text.slice(0, 300) || `HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const line = frame.split("\n").find((item) => item.startsWith("data: "));
      if (!line) continue;
      try {
        onEvent(JSON.parse(line.slice(6)) as StreamEvent);
      } catch {
        // 半截 JSON 不该让整轮对话崩掉，丢掉这一帧即可
      }
    }
  }
}
