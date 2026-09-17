/** 主窗口：左栏会话列表 + 右侧对话区 + 左下角「更多」（打开设置窗口）。
 *
 * 滚动是分开的：左栏列表滚自己的，右侧对话滚自己的，页面本身不滚，
 * 所以拖动一侧的滚动条不会带动另一侧。
 */
import { api, streamChat } from "./api";
import { Transcript } from "./chat";
import { Sidebar } from "./sidebar";
import type { Bootstrap, ConversationSummary, KbState, WorkspaceInfo } from "./types";
import "./styles.css";

function need<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`页面缺少节点 #${id}`);
  return node as T;
}

const transcript = new Transcript(need("transcript"));
const composer = need<HTMLTextAreaElement>("input");
const sendBtn = need<HTMLButtonElement>("send");
const modelTag = need("model-tag");
const kbTag = need("kb-tag");
const wsTag = need("ws-tag");

let conversations: ConversationSummary[] = [];
let currentConversation = "";
let busy = false;

const sidebar = new Sidebar(need("sidebar"), {
  onSelectConversation: (id) => void openConversation(id),
  onNewConversation: () => void newConversation(),
  onDeleteConversation: (id) => void deleteConversation(id),
  onOpenSettings: () => openSettingsWindow(),
});

//: 与 DSH 的辅助窗口一致：880×720 内容尺寸、居中、可缩放、不可最大化
const SETTINGS_WINDOW = { width: 880, height: 720 };

/**
 * 在独立窗口里打开设置——这是 DSH 里"设置按钮弹出新窗口"的做法：
 * 固定尺寸的辅助窗口、居中显示、不占用主窗口（所以主窗口的对话不会被打断）。
 * 浏览器里用 `window.open` 的弹窗特性复刻：popup + 精确尺寸 + 居中的 left/top +
 * 去掉菜单栏/工具栏/地址栏（Electron 那边对应 autoHideMenuBar + removeMenu）。
 * 窗口名固定，所以重复点「更多」只会聚焦同一个设置窗口，不会开一堆。
 */
function openSettingsWindow(): void {
  const { width, height } = SETTINGS_WINDOW;
  const screenX = window.screenX || window.screenLeft || 0;
  const screenY = window.screenY || window.screenTop || 0;
  const originWidth = window.outerWidth || width;
  const originHeight = window.outerHeight || height;
  const left = Math.max(0, Math.round(screenX + (originWidth - width) / 2));
  const top = Math.max(0, Math.round(screenY + (originHeight - height) / 2));
  const features = [
    "popup=yes",
    `width=${width}`,
    `height=${height}`,
    `left=${left}`,
    `top=${top}`,
    "menubar=no",
    "toolbar=no",
    "location=no",
    "status=no",
    "scrollbars=no",
    "resizable=yes",
  ].join(",");

  const win = window.open("/settings", "agent-settings", features);
  win?.focus();
}

function setBusy(value: boolean): void {
  busy = value;
  sendBtn.disabled = value;
  sendBtn.textContent = value ? "生成中…" : "发送";
}

function setConversations(items: ConversationSummary[], selected = currentConversation): void {
  conversations = items;
  sidebar.setConversations(items, selected);
}

function applyKb(state: KbState): void {
  kbTag.textContent = state.ready
    ? `知识库：${state.active_files} 个文档 · ${state.chunks} 块`
    : "知识库：未导入";
  kbTag.classList.toggle("bad", !state.ready);
}

function applyWorkspace(workspace: WorkspaceInfo): void {
  wsTag.textContent = `工作区：${workspace.name}`;
  wsTag.title = workspace.folder;
}

function welcome(): void {
  transcript.showWelcome(
    "开始一个新对话",
    "Agent 会自己判断要不要查知识库、要不要联网。左下角「更多」里可以配置模型和资料文件夹。",
  );
}

async function openConversation(id: string): Promise<void> {
  if (busy) return;
  try {
    const conversation = await api.getConversation(id);
    currentConversation = id;
    sidebar.setConversations(conversations, id);
    transcript.replay(conversation.messages);
    transcript.scrollToBottom();
  } catch (error) {
    console.error("读取会话失败", error);
  }
}

async function newConversation(): Promise<void> {
  if (busy) return;
  const result = await api.createConversation();
  currentConversation = result.conversation.id;
  setConversations(result.conversations, currentConversation);
  welcome();
  composer.focus();
}

async function deleteConversation(id: string): Promise<void> {
  if (busy) return;
  if (!window.confirm("删除这个对话？删除后无法恢复。")) return;
  const result = await api.deleteConversation(id);
  if (id === currentConversation) {
    currentConversation = "";
    const next = result.conversations[0];
    if (next) {
      await openConversation(next.id);
    } else {
      setConversations(result.conversations, "");
      welcome();
    }
    return;
  }
  setConversations(result.conversations, currentConversation);
}

async function ask(): Promise<void> {
  const question = composer.value.trim();
  if (!question || busy) return;

  setBusy(true);
  composer.value = "";
  transcript.addUser(question, new Date().toLocaleString("zh-CN", { hour12: false }));
  const turn = transcript.beginTurn();
  transcript.scrollToBottom();

  const choice = api.loadChoice();
  try {
    await streamChat(
      {
        question,
        conversation_id: currentConversation || null,
        provider: choice.provider,
        model: choice.model,
        api_key: choice.api_key,
        base_url: choice.base_url,
      },
      (event) => {
        if (event.type === "conversation") {
          currentConversation = event.id;
          return;
        }
        if (event.type === "saved") {
          void refreshConversations();
          return;
        }
        turn.apply(event);
        transcript.scrollToBottom();
      },
    );
  } catch (error) {
    turn.apply({ type: "error", message: (error as Error).message });
  } finally {
    setBusy(false);
    composer.focus();
  }
}

async function refreshConversations(): Promise<void> {
  try {
    const data = await api.bootstrap();
    setConversations(data.conversations, currentConversation);
    applyKb(data.kb);
    applyWorkspace(data.workspace);
  } catch (error) {
    console.error("刷新失败", error);
  }
}

sendBtn.addEventListener("click", () => void ask());
composer.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    void ask();
  }
});

// 设置窗口保存后通知主窗口；窗口重新获得焦点时也刷新一次，
// 这样即使 postMessage 丢了（换浏览器、窗口被拦截等）状态也是对的。
window.addEventListener("message", (event) => {
  if (event.origin !== window.location.origin) return;
  if ((event.data as { type?: string })?.type === "agent-settings-changed") {
    void refreshConversations();
    updateModelTag();
  }
});
window.addEventListener("focus", () => {
  updateModelTag();
  void refreshConversations();
});

function updateModelTag(): void {
  const choice = api.loadChoice();
  modelTag.textContent = choice.model || `${choice.provider}（默认模型）`;
}

async function bootstrap(): Promise<void> {
  let data: Bootstrap;
  try {
    data = await api.bootstrap();
  } catch (error) {
    modelTag.textContent = `初始化失败：${(error as Error).message}`;
    return;
  }
  updateModelTag();
  applyKb(data.kb);
  applyWorkspace(data.workspace);
  setConversations(data.conversations, "");
  if (data.conversations.length) {
    await openConversation(data.conversations[0].id);
  } else {
    welcome();
  }
}

void bootstrap();
composer.focus();
