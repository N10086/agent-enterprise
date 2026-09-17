/** 设置窗口：两个页签 —— API 配置 / 知识库。
 *
 * 独立窗口而不是页内弹层：配置是"另一件事"，占满一个窗口更好操作，
 * 也不会把主界面的对话挤变形。保存后通过 postMessage 通知主窗口刷新。
 */
import { api } from "./api";
import type { DirListing, KbState, ModelChoice, Provider } from "./types";
// 设置窗口有自己的一套样式（窗口底色、两栏布局都照 DSH 的辅助窗口来），
// 所以刻意不引主窗口的 styles.css，避免两套 token 互相污染。
import "./settings.css";

function need<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`页面缺少节点 #${id}`);
  return node as T;
}

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

function humanSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

function notifyMainWindow(): void {
  try {
    window.opener?.postMessage({ type: "agent-settings-changed" }, window.location.origin);
  } catch {
    // 主窗口可能已经关掉，忽略
  }
}

// ---------------------------------------------------------------- 页签

const PANE_TITLES: Record<string, string> = {
  api: "API 配置",
  kb: "知识库",
};

function setupTabs(): void {
  const buttons = Array.from(document.querySelectorAll<HTMLButtonElement>(".nav-item"));
  const panes = Array.from(document.querySelectorAll<HTMLElement>(".pane"));
  const title = need("pane-title");

  const activate = (name: string, remember = true): void => {
    for (const button of buttons) {
      button.classList.toggle("active", button.dataset.tab === name);
    }
    for (const pane of panes) {
      pane.hidden = pane.dataset.pane !== name;
    }
    title.textContent = PANE_TITLES[name] ?? name;
    if (remember) localStorage.setItem("agent-ui-settings-tab", name);
  };

  for (const button of buttons) {
    button.addEventListener("click", () => activate(button.dataset.tab ?? "api"));
  }
  activate(localStorage.getItem("agent-ui-settings-tab") ?? "api", false);

  // 独立窗口：给一个关闭按钮，行为与 DSH 辅助窗口右上角的关闭一致
  need("close-btn").addEventListener("click", () => window.close());
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") window.close();
  });
}

// ---------------------------------------------------------------- API 页签

let providers: Provider[] = [];

function setupApiTab(fallbackModel: string): void {
  const select = need<HTMLSelectElement>("provider");
  const modelInput = need<HTMLInputElement>("model");
  const keyInput = need<HTMLInputElement>("api-key");
  const baseInput = need<HTMLInputElement>("base-url");
  const hint = need("api-hint");
  const saveBtn = need<HTMLButtonElement>("save-api");
  const status = need("api-status");

  const current = (): Provider | undefined => providers.find((item) => item.key === select.value);

  const refreshHint = (): void => {
    const provider = current();
    if (keyInput.value.trim()) hint.textContent = "使用这里填写的 Key";
    else if (provider?.has_key) hint.textContent = `服务端已配置 ${provider.env_key}`;
    else hint.textContent = provider?.note || `请填写 ${provider?.env_key ?? "API Key"}`;
  };

  const applyProvider = (): void => {
    const provider = current();
    const datalist = need<HTMLDataListElement>("model-options");
    datalist.replaceChildren();
    for (const model of provider?.models ?? []) {
      const option = el("option");
      option.value = model;
      datalist.append(option);
    }
    modelInput.value = provider?.models[0] ?? "";
    baseInput.placeholder = provider?.base_url || "https://.../v1";
    baseInput.value = "";
    const saved = api.loadChoice();
    keyInput.value = saved.provider === select.value ? saved.api_key ?? "" : "";
    refreshHint();
  };

  select.addEventListener("change", applyProvider);
  keyInput.addEventListener("input", refreshHint);

  saveBtn.addEventListener("click", () => {
    const choice: ModelChoice = {
      provider: select.value,
      model: modelInput.value.trim(),
      api_key: keyInput.value.trim() || null,
      base_url: baseInput.value.trim() || null,
    };
    api.saveChoice(choice);
    status.textContent = `已保存：${current()?.label ?? ""} · ${choice.model || "默认模型"}`;
    status.classList.remove("bad");
    notifyMainWindow();
  });

  api
    .providers()
    .then((payload) => {
      providers = payload.providers;
      select.replaceChildren();
      for (const provider of providers) {
        const option = el("option");
        option.value = provider.key;
        option.textContent = provider.has_key ? `${provider.label} · 已配置` : provider.label;
        select.append(option);
      }
      const saved = api.loadChoice();
      select.value = providers.some((item) => item.key === saved.provider)
        ? saved.provider
        : providers[0]?.key ?? "";
      applyProvider();
      if (saved.model) modelInput.value = saved.model;
      else if (!current()?.models.length) modelInput.value = fallbackModel;
      status.textContent = saved.model ? `当前使用：${saved.model}` : "尚未保存选择，将使用默认模型";
    })
    .catch((error: Error) => {
      status.textContent = `读取厂商列表失败：${error.message}`;
      status.classList.add("bad");
    });
}

// ---------------------------------------------------------------- 知识库页签

function setupKbTab(): void {
  const folderTag = need("kb-folder");
  const fileList = need<HTMLUListElement>("kb-list");
  const status = need("kb-status");
  const drop = need("kb-drop");
  const fileInput = need<HTMLInputElement>("kb-file");
  const clearBtn = need<HTMLButtonElement>("kb-clear");

  // 文件夹浏览器
  const pathInput = need<HTMLInputElement>("folder-path");
  const dirList = need<HTMLUListElement>("dir-list");
  const useBtn = need<HTMLButtonElement>("use-folder");

  let busy = false;

  const render = (state: KbState): void => {
    folderTag.textContent = state.folder;
    folderTag.title = state.is_default_folder ? "默认工作区（应用管理的文件夹）" : state.folder;
    pathInput.value = state.folder;

    fileList.replaceChildren();
    for (const file of state.files) {
      const row = el("li", "kb-item");
      row.classList.toggle("excluded", file.excluded);

      const info = el("div", "kb-meta");
      info.append(
        el("span", "kb-name", file.name),
        el(
          "span",
          "kb-sub muted small",
          file.excluded
            ? `${humanSize(file.size)} · 已移出知识库（原文件保留）`
            : `${humanSize(file.size)} · ${file.chunks || "未索引"} 段`,
        ),
      );

      const toggle = el("button", "kb-del", file.excluded ? "＋" : "✕");
      toggle.type = "button";
      toggle.title = file.excluded ? "重新纳入知识库" : "移出知识库（不删除原文件）";
      toggle.addEventListener("click", () => void toggleFile(file.name, file.excluded));

      row.append(info, toggle);
      fileList.append(row);
    }

    if (!state.files.length) {
      fileList.append(el("li", "kb-empty muted", "这个文件夹里还没有可用的文档"));
    }

    clearBtn.hidden = !state.files.some((file) => !file.excluded);
    if (state.ready) {
      status.textContent = `${state.active_files} 个文档参与检索 · ${state.chunks} 个片段 · 构建于 ${state.built_at}`;
      status.classList.remove("bad");
    } else if (state.files.length) {
      status.textContent = "当前没有文档参与检索（都被移出了？）";
      status.classList.add("bad");
    } else {
      status.textContent = "导入文档后，Agent 才会在回答时检索它们";
    }
  };

  const toggleFile = async (name: string, excluded: boolean): Promise<void> => {
    if (busy) return;
    busy = true;
    status.textContent = excluded ? `正在重新纳入 ${name}…` : `正在移出 ${name}…`;
    try {
      const result = excluded ? await api.includeKbFile(name) : await api.excludeKbFile(name);
      if (result.kb) render(result.kb);
      notifyMainWindow();
    } catch (error) {
      status.textContent = `操作失败：${(error as Error).message}`;
      status.classList.add("bad");
    } finally {
      busy = false;
    }
  };

  const upload = async (files: File[]): Promise<void> => {
    if (busy || !files.length) return;
    busy = true;
    status.textContent = `正在解析 ${files.length} 个文件…`;
    status.classList.remove("bad");
    try {
      const payload = await Promise.all(
        files.map(async (file) => ({ name: file.name, data: await api.readAsBase64(file) })),
      );
      const result = await api.uploadKb(payload);
      if (!result.ok) throw new Error(result.error ?? "导入失败");
      if (result.kb) render(result.kb);
      const parts = [`已导入 ${result.files?.length ?? 0} 个文件`];
      if (result.failures?.length) {
        parts.push(`${result.failures.length} 个失败：${result.failures[0].error}`);
        status.classList.add("bad");
      }
      status.textContent = parts.join(" · ");
      fileInput.value = "";
      notifyMainWindow();
    } catch (error) {
      status.textContent = `导入失败：${(error as Error).message}`;
      status.classList.add("bad");
    } finally {
      busy = false;
    }
  };

  drop.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    if (fileInput.files?.length) void upload(Array.from(fileInput.files));
  });
  for (const name of ["dragenter", "dragover"] as const) {
    drop.addEventListener(name, (event) => {
      event.preventDefault();
      drop.classList.add("over");
    });
  }
  for (const name of ["dragleave", "drop"] as const) {
    drop.addEventListener(name, (event) => {
      event.preventDefault();
      drop.classList.remove("over");
    });
  }
  drop.addEventListener("drop", (event) => {
    const files = Array.from(event.dataTransfer?.files ?? []);
    if (files.length) void upload(files);
  });

  clearBtn.addEventListener("click", () => {
    if (!window.confirm("把所有文档移出知识库？（不会删除磁盘上的原文件）")) return;
    void (async () => {
      const result = await api.clearKb();
      if (result.kb) render(result.kb);
      status.textContent = "已全部移出知识库（原文件保留在文件夹里）";
      notifyMainWindow();
    })();
  });

  const browse = async (path: string): Promise<void> => {
    try {
      const listing: DirListing = await api.listDir(path);
      pathInput.value = listing.path;
      dirList.replaceChildren();
      if (listing.parent) {
        const up = el("li", "dir-item up", "↑ 上一级");
        up.addEventListener("click", () => void browse(listing.parent as string));
        dirList.append(up);
      }
      for (const dir of listing.dirs) {
        const item = el("li", "dir-item", `📁 ${dir.name}`);
        item.title = dir.path;
        item.addEventListener("click", () => void browse(dir.path));
        dirList.append(item);
      }
      if (listing.is_workspace) {
        dirList.prepend(el("li", "dir-item current muted", "（当前工作区）"));
      }
      status.textContent = `${listing.path}：${listing.dirs.length} 个子目录、${listing.files} 个文件`;
    } catch (error) {
      status.textContent = `打开目录失败：${(error as Error).message}`;
      status.classList.add("bad");
    }
  };

  useBtn.addEventListener("click", () => {
    const path = pathInput.value.trim();
    if (!path) return;
    void (async () => {
      busy = true;
      status.textContent = "正在切换工作区并重建索引…";
      try {
        const result = await api.setFolder(path);
        render(result.kb);
        await browse(result.workspace.folder);
        status.textContent = `工作区已切换到 ${result.workspace.folder}`;
        notifyMainWindow();
      } catch (error) {
        status.textContent = `切换失败：${(error as Error).message}`;
        status.classList.add("bad");
      } finally {
        busy = false;
      }
    })();
  });

  api
    .bootstrap()
    .then((data) => {
      render(data.kb);
      void browse(data.workspace.folder);
    })
    .catch((error: Error) => {
      status.textContent = `读取知识库状态失败：${error.message}`;
      status.classList.add("bad");
    });
}

// ---------------------------------------------------------------- 启动

setupTabs();
api
  .bootstrap()
  .then((data) => {
    setupApiTab(data.model);
    setupKbTab();
  })
  .catch((error: Error) => {
    need("api-status").textContent = `初始化失败：${error.message}`;
  });
