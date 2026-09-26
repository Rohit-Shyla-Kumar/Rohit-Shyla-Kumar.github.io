import { createFileRoute } from "@tanstack/react-router";
import { PageHeader } from "@/components/page-header";
import { SkillBoard } from "@/components/skill-board";
import { profile } from "@/content/profile";
import { stack } from "@/content/skills";

export const Route = createFileRoute("/skills")({ component: Skills });

function Skills() {
  return (
    <div>
      <PageHeader
        title="Skills"
        lede="A self-score against real production work — not a certificate exam. Pick a group to see the detail."
      />
      <SkillBoard />

      <section className="mt-12 grid gap-6 lg:grid-cols-2">
        <div className="panel p-6">
          <h2 className="font-display text-lg tracking-tight">Certifications</h2>
          <ul className="mt-4 space-y-3 text-sm">
            {profile.certifications.map((cert) => (
              <li key={cert.name}>
                <p className="text-fg">{cert.name}</p>
                <p className="text-muted">
                  {cert.detail} · {cert.date}
                </p>
              </li>
            ))}
          </ul>
        </div>
        <div className="panel p-6">
          <h2 className="font-display text-lg tracking-tight">Awards</h2>
          <ul className="mt-4 space-y-3 text-sm text-muted">
            {profile.awards.map((award) => (
              <li key={award}>{award}</li>
            ))}
          </ul>
        </div>
      </section>

      <section className="mt-8">
        <p className="mb-3 text-xs tracking-widest text-muted uppercase">Tools</p>
        <div className="flex flex-wrap gap-2">
          {stack.map((item) => (
            <span key={item} className="rounded-md border border-line px-2 py-1 text-xs text-muted">
              {item}
            </span>
          ))}
        </div>
      </section>
    </div>
  );
}
