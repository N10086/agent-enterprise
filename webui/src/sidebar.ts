/** 左侧会话栏：新建 / 切换 / 删除对话，左下角「更多」打开设置窗口。
 *
 * 左栏只放会话——配置类的东西（API、知识库）都在独立的设置窗口里，
 * 这样对话列表能一直占满左栏，也不会被表单挤开。
 */
import type { ConversationSummary } from "./types";

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export interface SidebarCallbacks {
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  onDeleteConversation: (id: string) => void;
  onOpenSettings: () => void;
}

export class Sidebar {
  private readonly list: HTMLUListElement;
  private readonly countTag: HTMLSpanElement;

  constructor(root: HTMLElement, private readonly cb: SidebarCallbacks) {
    root.innerHTML = "";

    const actions = el("div", "ws-actions");
    const newBtn = el("button", "primary-btn", "＋ 新对话");
    newBtn.type = "button";
    newBtn.addEventListener("click", () => cb.onNewConversation());
    actions.append(newBtn);
    root.append(actions);

    const listHead = el("div", "list-head");
    listHead.append(el("span", "section-title", "会话"));
    this.countTag = el("span", "muted small");
    listHead.append(this.countTag);
    root.append(listHead);

    this.list = el("ul", "conv-list");
    root.append(this.list);

    const footer = el("div", "side-footer");
    const more = el("button", "ghost-btn", "⋯ 更多");
    more.type = "button";
    more.title = "打开设置窗口（API 配置 / 知识库）";
    more.addEventListener("click", () => cb.onOpenSettings());
    footer.append(more);
    root.append(footer);
  }

  setConversations(items: ConversationSummary[], selected: string): void {
    this.countTag.textContent = `${items.length} 个`;
    this.list.replaceChildren();
    if (!items.length) {
      this.list.append(el("li", "conv-empty muted", "还没有会话，点上面的「新对话」"));
      return;
    }
    for (const item of items) {
      const row = el("li", "conv-item");
      row.classList.toggle("active", item.id === selected);

      const title = el("button", "conv-title", item.title || "未命名");
      title.type = "button";
      title.title = `${item.title}\n${item.updated_at}`;
      title.addEventListener("click", () => this.cb.onSelectConversation(item.id));

      const remove = el("button", "conv-del", "✕");
      remove.type = "button";
      remove.title = "删除这个对话";
      remove.addEventListener("click", (event) => {
        event.stopPropagation();
        this.cb.onDeleteConversation(item.id);
      });

      row.append(title, remove);
      this.list.append(row);
    }
  }
}
