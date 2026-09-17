/** 与后端接口对齐的类型定义。字段名保持 snake_case，避免两边各起一套名字。 */

export interface Provider {
  key: string;
  label: string;
  base_url: string;
  models: string[];
  has_key: boolean;
  env_key: string;
  note: string;
}

export interface ProvidersPayload {
  providers: Provider[];
  default: { provider: string; model: string };
}

export interface KbFile {
  name: string;
  size: number;
  excluded: boolean;
  chars: number;
  chunks: number;
}

export interface KbState {
  mode: string;
  ready: boolean;
  folder: string;
  folder_name: string;
  is_default_folder: boolean;
  files: KbFile[];
  active_files: number;
  chunks: number;
  built_at: string;
}

export interface UploadResult {
  ok: boolean;
  error?: string;
  files?: { name: string; chars: number; chunks: number }[];
  failures?: { name: string; error: string }[];
  chunks?: number;
  kb?: KbState;
}

export interface WorkspaceInfo {
  folder: string;
  name: string;
  is_default: boolean;
  files: number;
  excluded: string[];
}

export interface DirEntry {
  name: string;
  path: string;
}

export interface DirListing {
  path: string;
  parent: string | null;
  dirs: DirEntry[];
  files: number;
  is_workspace: boolean;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: number;
}

/** 落盘的一条历史消息（助手消息带执行轨迹，重新打开页面也能完整回放）。 */
export interface StoredMessage {
  role: "user" | "assistant";
  text: string;
  at?: string;
  steps?: string[];
  tools?: { name: string; args: Record<string, unknown> }[];
  sources?: { source: string; title: string }[];
  usage?: Usage;
  seconds?: number;
  status?: string;
  error?: string;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: StoredMessage[];
}

export interface Bootstrap {
  model: string;
  providers: ProvidersPayload;
  workspace: WorkspaceInfo;
  conversations: ConversationSummary[];
  kb: KbState;
  mcp: string;
  tools: string[];
  grade_enabled: boolean;
  grade_score: number;
  top_k: number;
}

export interface Usage {
  llm_calls: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
}

/** 后端 SSE 推来的事件；type 决定其余字段，逐个收窄。 */
export type StreamEvent =
  | { type: "conversation"; id: string }
  | { type: "node"; node: string; label: string }
  | { type: "tool"; name: string; args: Record<string, unknown> }
  | { type: "sources"; items: { source: string; title: string }[] }
  | { type: "grade"; grade: string; note: string }
  | {
      type: "answer";
      text: string;
      status?: string;
      verification?: string;
      tools_used?: number;
      review_notes?: string;
    }
  | { type: "usage"; llm_calls: number; input_tokens: number; output_tokens: number; total_tokens: number }
  | { type: "done"; seconds: number; warning?: string }
  | { type: "saved"; conversation_id: string; title: string }
  | { type: "error"; message: string }
  | { type: "end" };

export interface ChatRequest {
  question: string;
  conversation_id: string | null;
  provider: string;
  model: string;
  api_key: string | null;
  base_url: string | null;
}

/** 模型选择存在浏览器本地，主窗口与设置窗口共用。 */
export interface ModelChoice {
  provider: string;
  model: string;
  api_key: string | null;
  base_url: string | null;
}
