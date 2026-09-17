/** 对话区：用户提问、Agent 执行过程（步骤 / 工具 / 来源）、最终答案。
 *
 * 两个入口都用同一套渲染：
 *   - `Turn.apply(event)` —— 实时流式推进（后端 SSE 推什么就画什么）；
 *   - `Turn.replay(message)` —— 从落盘的历史消息还原（重新打开页面时用）。
 */
import { compactJSON, renderMarkdown } from "./markdown";
import type { StoredMessage, StreamEvent, Usage } from "./types";

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

class Step {
  readonly root: HTMLLIElement;
  private readonly dot: HTMLSpanElement;
  private readonly chips: HTMLDivElement;

  constructor(label: string) {
    this.root = el("li", "step");
    this.dot = el("span", "dot");
    this.dot.dataset.state = "ongoing";
    this.root.append(this.dot, el("span", "step-label", label));
    this.chips = el("div", "step-chips");
    this.root.append(this.chips);
  }

  finish(): void {
    this.dot.dataset.state = "done";
  }

  addChip(text: string, kind?: string): void {
    this.chips.append(el("span", kind ? `chip ${kind}` : "chip", text));
  }
}

export class Turn {
  readonly root: HTMLElement;
  private readonly title: HTMLSpanElement;
  private readonly steps: HTMLOListElement;
  private readonly answer: HTMLDivElement;
  private readonly foot: HTMLDivElement;
  private sourcesBox: HTMLDetailsElement | null = null;
  private current: Step | null = null;
  private finished = false;

  constructor() {
    this.root = el("article", "turn");
    const head = el("div", "turn-head");
    const dot = el("span", "dot");
    dot.dataset.state = "ongoing";
    this.title = el("span", "turn-title", "正在处理…");
    head.append(dot, this.title);
    this.root.append(head);

    this.steps = el("ol", "steps");
    this.answer = el("div", "answer");
    this.foot = el("div", "turn-foot");
    this.root.append(this.steps, this.answer, this.foot);
  }

  private step(label: string): Step {
    this.current?.finish();
    const step = new Step(label);
    this.steps.append(step.root);
    this.current = step;
    return step;
  }

  apply(event: StreamEvent): void {
    switch (event.type) {
      case "node":
        this.step(event.label || event.node);
        this.title.textContent = "执行中…";
        break;
      case "tool": {
        const target = this.current ?? this.step("工具调用");
        target.addChip(`🔧 ${event.name} ${compactJSON(event.args)}`, "tool");
        break;
      }
      case "grade": {
        const target = this.current ?? this.step("检索质检");
        const bad = event.grade === "irrelevant";
        target.addChip(bad ? "不相关 → 换查询重检" : "相关", bad ? "bad" : "ok");
        break;
      }
      case "sources":
        this.renderSources(event.items);
        break;
      case "answer":
        this.answer.innerHTML = event.text
          ? renderMarkdown(event.text)
          : '<p class="muted">（没有产出内容）</p>';
        if (event.review_notes) this.foot.append(el("span", "chip", `审核：${event.review_notes}`));
        break;
      case "usage":
        this.renderUsage(event);
        break;
      case "done":
        this.finish(event.seconds, event.warning);
        break;
      case "error": {
        this.root.classList.add("failed");
        this.foot.append(el("span", "chip bad", `错误：${event.message}`));
        this.finish();
        break;
      }
      default:
        break;
    }
  }

  /** 用落盘的消息还原一个已完成的回合。 */
  replay(message: StoredMessage): void {
    for (const label of message.steps ?? []) {
      const step = this.step(label);
      step.finish();
    }
    for (const tool of message.tools ?? []) {
      const target = this.current ?? this.step("工具调用");
      target.addChip(`🔧 ${tool.name} ${compactJSON(tool.args)}`, "tool");
    }
    if (message.sources?.length) this.renderSources(message.sources);
    this.answer.innerHTML = message.text
      ? renderMarkdown(message.text)
      : '<p class="muted">（没有产出内容）</p>';
    if (message.usage?.total_tokens) this.renderUsage(message.usage);
    if (message.error) {
      this.root.classList.add("failed");
      this.foot.append(el("span", "chip bad", `错误：${message.error}`));
    }
    this.finish(message.seconds);
    if (message.at) this.foot.append(el("span", "chip", message.at));
  }

  private renderSources(items: { source: string; title: string }[]): void {
    if (!this.sourcesBox) {
      this.sourcesBox = el("details", "sources");
      this.sourcesBox.append(el("summary", undefined, "检索到的片段"), el("ul"));
      this.root.insertBefore(this.sourcesBox, this.answer);
    }
    const list = this.sourcesBox.querySelector("ul")!;
    for (const item of items) {
      const text = item.source ? `${item.title}（${item.source}）` : item.title;
      if (!Array.from(list.children).some((node) => node.textContent === text)) {
        list.append(el("li", undefined, text));
      }
    }
  }

  private renderUsage(usage: Usage): void {
    this.foot.append(
      el(
        "span",
        "chip",
        `${usage.llm_calls} 次调用 · ${usage.total_tokens} token（入 ${usage.input_tokens} / 出 ${usage.output_tokens}）`,
      ),
    );
  }

  finish(seconds?: number, warning?: string): void {
    if (this.finished) return;
    this.finished = true;
    this.current?.finish();
    this.root.querySelectorAll<HTMLElement>(".turn-head .dot").forEach((node) => {
      node.dataset.state = warning ? "warn" : "done";
    });
    this.title.textContent = warning ? "已完成（有告警）" : "已完成";
    if (seconds !== undefined) this.foot.append(el("span", "chip", `耗时 ${seconds}s`));
    if (warning) this.foot.append(el("span", "chip bad", warning));
  }
}

export class Transcript {
  constructor(private readonly root: HTMLElement) {}

  showWelcome(text: string, hint: string): void {
    const box = el("div", "welcome");
    const title = el("h2", undefined, text);
    const note = el("p", undefined, hint);
    box.append(title, note);
    this.root.replaceChildren(box);
  }

  private dropWelcome(): void {
    this.root.querySelector(".welcome")?.remove();
  }

  addUser(text: string, at?: string): void {
    this.dropWelcome();
    const node = el("div", "msg user");
    node.append(el("div", "msg-body", text));
    if (at) node.append(el("div", "msg-time", at));
    this.root.append(node);
  }

  beginTurn(): Turn {
    this.dropWelcome();
    const turn = new Turn();
    this.root.append(turn.root);
    return turn;
  }

  replay(messages: StoredMessage[]): void {
    this.root.replaceChildren();
    if (!messages.length) {
      this.showWelcome("这个对话还没有内容", "在下面输入问题开始吧。");
      return;
    }
    for (const message of messages) {
      if (message.role === "user") {
        this.addUser(message.text, message.at);
      } else {
        this.dropWelcome();
        const turn = new Turn();
        this.root.append(turn.root);
        turn.replay(message);
      }
    }
  }

  scrollToBottom(): void {
    this.root.scrollTop = this.root.scrollHeight;
  }
}
