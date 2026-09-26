import { createFileRoute } from "@tanstack/react-router";
import { Download, Printer } from "lucide-react";
import { PageHeader } from "@/components/page-header";
import { roles } from "@/content/experience";
import { profile } from "@/content/profile";
import { stack } from "@/content/skills";

export const Route = createFileRoute("/resume")({ component: Resume });

function Resume() {
  return (
    <div>
      <PageHeader
        title="Resume"
        lede="Download a copy, or print this page to a two-page PDF."
      />
      <div className="no-print mb-8 flex flex-wrap gap-3">
        <a
          href="/Rohit-Shyla-Kumar-Resume.docx"
          download
          className="inline-flex min-h-11 items-center gap-2 rounded-md bg-fg px-4 text-sm text-bg hover:opacity-90 active:scale-[0.96]"
        >
          <Download className="size-4" />
          Download Word
        </a>
        <a
          href="/Rohit-Shyla-Kumar-Resume.txt"
          download
          className="inline-flex min-h-11 items-center gap-2 rounded-md border border-line px-4 text-sm hover:border-fg active:scale-[0.96]"
        >
          <Download className="size-4" />
          Download text
        </a>
        <button
          type="button"
          onClick={() => window.print()}
          className="inline-flex min-h-11 items-center gap-2 rounded-md border border-line px-4 text-sm text-muted hover:border-fg hover:text-fg active:scale-[0.96]"
        >
          <Printer className="size-4" />
          Print / save PDF
        </button>
      </div>

      <article className="resume-sheet panel max-w-3xl p-6 sm:p-8">
        <header className="border-b border-line pb-3">
          <h2 className="font-display text-2xl tracking-tight">{profile.name}</h2>
          <p className="mt-1 text-sm text-muted">{profile.tagline}</p>
          <p className="mt-1 text-xs text-dim">
            {profile.email} · github.com/Rohit-Shyla-Kumar · hk.linkedin.com/in/rohit-shyla-kumar
          </p>
        </header>

        <section className="mt-4">
          <h3 className="text-xs tracking-widest text-muted uppercase">Summary</h3>
          <p className="mt-1 text-sm leading-snug text-pretty">{profile.summary}</p>
        </section>

        <section className="mt-4">
          <h3 className="text-xs tracking-widest text-muted uppercase">Experience</h3>
          <div className="mt-2 space-y-3">
            {roles.map((role) => (
              <div key={role.id}>
                <div className="flex flex-wrap justify-between gap-2 text-sm">
                  <p>
                    {role.title} · {role.org}
                  </p>
                  <p className="tabular-nums text-muted">{role.dates}</p>
                </div>
                <ul className="mt-1 space-y-0.5 text-sm text-fg/90">
                  {role.bullets.map((b) => (
                    <li key={b} className="flex gap-2">
                      <span className="mt-2 size-1 shrink-0 rounded-full bg-fg" />
                      <span>{b}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>

        <section className="mt-4">
          <h3 className="text-xs tracking-widest text-muted uppercase">Skills</h3>
          <p className="mt-1 text-sm text-fg/90">{stack.join(" · ")}</p>
        </section>

        <section className="mt-4">
          <h3 className="text-xs tracking-widest text-muted uppercase">Education & certs</h3>
          <p className="mt-1 text-sm">
            {profile.education.degree}, {profile.education.school} ({profile.education.dates})
          </p>
          <p className="text-sm text-muted">{profile.education.notes}</p>
          <ul className="mt-1 space-y-0.5 text-sm">
            {profile.certifications.map((c) => (
              <li key={c.name}>
                {c.name} — {c.detail} ({c.date})
              </li>
            ))}
          </ul>
        </section>

        <section className="mt-4">
          <h3 className="text-xs tracking-widest text-muted uppercase">Awards</h3>
          <ul className="mt-1 space-y-0.5 text-sm text-fg/90">
            {profile.awards.map((a) => (
              <li key={a}>{a}</li>
            ))}
          </ul>
        </section>
      </article>
    </div>
  );
}
