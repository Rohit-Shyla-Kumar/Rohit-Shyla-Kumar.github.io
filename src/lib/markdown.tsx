import type { ReactNode } from "react";

function inline(text: string, keyPrefix: string): ReactNode[] {
  const parts: ReactNode[] = [];
  const pattern =
    /(\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\)|`[^`]+`|\*[^*]+\*)/g;
  let last = 0;
  let match: RegExpExecArray | null;
  let i = 0;
  while ((match = pattern.exec(text))) {
    if (match.index > last) {
      parts.push(text.slice(last, match.index));
    }
    const token = match[0];
    if (token.startsWith("**")) {
      parts.push(
        <strong key={`${keyPrefix}-b-${i}`} className="font-medium text-fg">
          {token.slice(2, -2)}
        </strong>,
      );
    } else if (token.startsWith("[")) {
      const link = /\[([^\]]+)\]\(([^)]+)\)/.exec(token);
      if (link) {
        const href = link[2];
        const external = href.startsWith("http");
        parts.push(
          <a
            key={`${keyPrefix}-a-${i}`}
            href={href}
            className="underline decoration-fg/30 underline-offset-4 hover:decoration-fg"
            {...(external ? { target: "_blank", rel: "noreferrer" } : {})}
          >
            {link[1]}
          </a>,
        );
      }
    } else if (token.startsWith("`")) {
      parts.push(
        <code
          key={`${keyPrefix}-c-${i}`}
          className="rounded-xs border border-line bg-raised px-1 text-sm"
        >
          {token.slice(1, -1)}
        </code>,
      );
    } else {
      parts.push(
        <em key={`${keyPrefix}-i-${i}`} className="italic text-fg">
          {token.slice(1, -1)}
        </em>,
      );
    }
    last = match.index + token.length;
    i += 1;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

export function Markdown({ source }: { source: string }) {
  const blocks = source.trim().split(/\n{2,}/);
  return (
    <div className="space-y-5">
      {blocks.map((block, idx) => {
        const lines = block.split("\n");
        if (lines[0].startsWith("## ")) {
          return (
            <h2 key={idx} className="font-display text-lg tracking-tight">
              {lines[0].slice(3)}
            </h2>
          );
        }
        if (lines.every((line) => line.startsWith("- "))) {
          return (
            <ul key={idx} className="space-y-2 pl-1">
              {lines.map((line, i) => (
                <li key={i} className="flex gap-3 text-pretty">
                  <span className="mt-2 size-1 shrink-0 rounded-full bg-fg" />
                  <span>{inline(line.slice(2), `${idx}-${i}`)}</span>
                </li>
              ))}
            </ul>
          );
        }
        if (lines[0].startsWith("```")) {
          const code = lines.slice(1).join("\n").replace(/```$/, "");
          return (
            <pre
              key={idx}
              className="overflow-x-auto rounded-md border border-line bg-raised p-4 text-sm"
            >
              <code>{code}</code>
            </pre>
          );
        }
        return (
          <p key={idx} className="text-pretty leading-relaxed text-fg/90">
            {lines.map((line, i) => (
              <span key={i}>
                {inline(line, `${idx}-${i}`)}
                {i < lines.length - 1 ? " " : null}
              </span>
            ))}
          </p>
        );
      })}
    </div>
  );
}
