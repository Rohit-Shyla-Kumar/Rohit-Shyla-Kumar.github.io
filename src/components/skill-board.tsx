import { useEffect, useState } from "react";
import { domains, type Domain } from "@/content/skills";
import { cn } from "@/lib/utils";

function SkillBars({ domain }: { domain: Domain }) {
  return (
    <div className="space-y-5">
      {domain.skills.map((skill) => (
        <div key={skill.name}>
          <div className="mb-2 flex items-baseline justify-between gap-3 text-sm">
            <span>{skill.name}</span>
            <span className="tabular-nums text-muted">{skill.level}</span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-raised">
            <div
              className="skill-bar-fill h-full rounded-full bg-fg"
              style={{ width: `${skill.level}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

export function SkillBoard() {
  const [selected, setSelected] = useState(domains[0]!.id);
  const current = domains.find((d) => d.id === selected) ?? domains[0]!;

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      const idx = domains.findIndex((d) => d.id === selected);
      if (e.key === "ArrowDown" || e.key === "j") {
        e.preventDefault();
        const next = domains[(idx + 1) % domains.length];
        if (next) setSelected(next.id);
      }
      if (e.key === "ArrowUp" || e.key === "k") {
        e.preventDefault();
        const next = domains[(idx - 1 + domains.length) % domains.length];
        if (next) setSelected(next.id);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selected]);

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,16rem)_1fr]">
      <div className="panel overflow-hidden p-1">
        <ul>
          {domains.map((domain) => {
            const active = domain.id === selected;
            return (
              <li key={domain.id}>
                <button
                  type="button"
                  onClick={() => setSelected(domain.id)}
                  className={cn(
                    "flex w-full min-h-12 items-center px-4 text-left text-sm transition-colors duration-150",
                    active
                      ? "bg-fg text-bg"
                      : "text-muted hover:bg-raised hover:text-fg",
                  )}
                >
                  {domain.title}
                </button>
              </li>
            );
          })}
        </ul>
      </div>

      <div className="panel p-6 sm:p-8">
        <h2 className="font-display text-2xl tracking-tight">{current.title}</h2>
        <p className="mt-3 max-w-prose text-pretty leading-relaxed text-muted">
          {current.blurb}
        </p>
        <div className="mt-8">
          <SkillBars key={current.id} domain={current} />
        </div>
      </div>
    </div>
  );
}
