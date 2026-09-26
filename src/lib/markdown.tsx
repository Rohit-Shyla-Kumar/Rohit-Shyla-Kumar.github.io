import type { ReactNode } from "react";

function youtubeId(src: string): string | null {
  const trimmed = src.trim();
  const match =
    trimmed.match(/(?:youtu\.be\/|youtube\.com\/(?:watch\?v=|embed\/|shorts\/))([A-Za-z0-9_-]{11})/) ??
    trimmed.match(/^([A-Za-z0-9_-]{11})$/);
  return match ? match[1] : null;
}

function isVideoSrc(src: string) {
  return /\.(mp4|webm|ogg)(\?.*)?$/i.test(src);
}

function MediaFigure({
  caption,
  children,
}: {
  caption: string;
  children: ReactNode;
}) {
  return (
    <figure className="overflow-hidden rounded-md border border-line bg-raised">
      {children}
      {caption ? (
        <figcaption className="px-3 py-2 text-sm leading-relaxed text-muted">
          {caption}
        </figcaption>
      ) : null}
    </figure>
  );
}

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

function renderBlock(block: string, idx: number) {
  const trimmed = block.trim();
  const lines = trimmed.split("\n").map((line) => line.trimEnd());

  if (lines[0].startsWith("# ") && !lines[0].startsWith("## ")) {
    return (
      <h2 key={idx} className="font-display text-xl tracking-tight">
        {lines[0].slice(2)}
      </h2>
    );
  }
  if (lines[0].startsWith("## ")) {
    return (
      <h2 key={idx} className="font-display text-lg tracking-tight">
        {lines[0].slice(3)}
      </h2>
    );
  }

  const youtube = /^!youtube\[([^\]]*)\]\(([^)]+)\)$/.exec(trimmed);
  if (youtube) {
    const id = youtubeId(youtube[2]);
    if (id) {
      return (
        <MediaFigure key={idx} caption={youtube[1]}>
          <div className="aspect-video bg-black">
            <iframe
              className="size-full"
              src={`https://www.youtube.com/embed/${id}`}
              title={youtube[1] || "YouTube video"}
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            />
          </div>
        </MediaFigure>
      );
    }
  }

  const taggedVideo = /^!video\[([^\]]*)\]\(([^)]+)\)$/.exec(trimmed);
  if (taggedVideo) {
    return (
      <MediaFigure key={idx} caption={taggedVideo[1]}>
        <video className="w-full" controls playsInline src={taggedVideo[2]} />
      </MediaFigure>
    );
  }

  const image = /^!\[([^\]]*)\]\(([^)]+)\)$/.exec(trimmed);
  if (image) {
    const [, alt, src] = image;
    if (isVideoSrc(src)) {
      return (
        <MediaFigure key={idx} caption={alt}>
          <video className="w-full" controls playsInline src={src} />
        </MediaFigure>
      );
    }
    return (
      <MediaFigure key={idx} caption={alt}>
        <img src={src} alt={alt} className="w-full" />
      </MediaFigure>
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

  if (lines.every((line) => /^\d+\. /.test(line))) {
    return (
      <ol key={idx} className="list-decimal space-y-2 pl-6">
        {lines.map((line, i) => (
          <li key={i} className="text-pretty">
            {inline(line.replace(/^\d+\. /, ""), `${idx}-${i}`)}
          </li>
        ))}
      </ol>
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
          {inline(line.trim(), `${idx}-${i}`)}
          {i < lines.length - 1 ? " " : null}
        </span>
      ))}
    </p>
  );
}

export function Markdown({ source }: { source: string }) {
  const blocks = source.trim().split(/\n{2,}/);
  return <div className="space-y-5">{blocks.map((block, idx) => renderBlock(block, idx))}</div>;
}
