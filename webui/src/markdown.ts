/** 极简 Markdown 渲染。
 *
 * 刻意不做完整解析、也不引第三方库：模型输出一律先转义再套少量规则，
 * 浏览器里不会执行到模型给的 HTML。够用的范围是粗体、行内代码、
 * 无序/有序列表和段落换行——问答场景里 90% 的排版就这些。
 */

function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (char) => {
    switch (char) {
      case "&":
        return "&amp;";
      case "<":
        return "&lt;";
      case ">":
        return "&gt;";
      case '"':
        return "&quot;";
      default:
        return "&#39;";
    }
  });
}

function inline(text: string): string {
  return text
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(?<![*\w])\*([^*\n]+)\*(?!\*)/g, "<em>$1</em>");
}

export function renderMarkdown(source: string): string {
  const lines = escapeHtml(source ?? "").split("\n");
  const html: string[] = [];
  let listType: "ul" | "ol" | null = null;

  const closeList = (): void => {
    if (listType) {
      html.push(`</${listType}>`);
      listType = null;
    }
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (!line.trim()) {
      closeList();
      continue;
    }
    const bullet = /^\s*[-*+]\s+(.*)$/.exec(line);
    const numbered = /^\s*\d+[.)]\s+(.*)$/.exec(line);
    if (bullet || numbered) {
      const want: "ul" | "ol" = bullet ? "ul" : "ol";
      if (listType !== want) {
        closeList();
        html.push(`<${want}>`);
        listType = want;
      }
      html.push(`<li>${inline((bullet ?? numbered)![1])}</li>`);
      continue;
    }
    closeList();
    html.push(`<p>${inline(line)}</p>`);
  }
  closeList();
  return html.join("");
}

/** 工具入参这类短 JSON 的展示：压缩空白，过长截断。 */
export function compactJSON(value: unknown, limit = 90): string {
  let text: string;
  try {
    text = JSON.stringify(value);
  } catch {
    text = String(value);
  }
  text = text ?? "";
  return text.length > limit ? `${text.slice(0, limit)}…` : text;
}
