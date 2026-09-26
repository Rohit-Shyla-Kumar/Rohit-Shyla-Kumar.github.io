import { createFileRoute, Link } from "@tanstack/react-router";
import { PageHeader } from "@/components/page-header";
import { profile } from "@/content/profile";

export const Route = createFileRoute("/about")({ component: About });

function About() {
  return (
    <article>
      <PageHeader title="About" lede="A short version of how I work, and what I care about." />
      <div className="stagger-in grid gap-10 lg:grid-cols-[1fr_16rem]">
        <div className="max-w-prose space-y-5 text-fg/90 leading-relaxed">
          {profile.about.map((para) => (
            <p key={para} className="text-pretty">
              {para}
            </p>
          ))}
          <p className="text-pretty">
            See{" "}
            <Link to="/skills" className="underline underline-offset-4 hover:text-fg">
              skills
            </Link>{" "}
            for how I work, or the{" "}
            <Link to="/resume" className="underline underline-offset-4 hover:text-fg">
              resume
            </Link>{" "}
            for the paper trail.
          </p>
        </div>
        <aside className="space-y-6">
          <div className="panel p-5 text-sm">
            <p className="text-xs tracking-widest text-muted uppercase">Education</p>
            <p className="mt-3 text-fg">{profile.education.school}</p>
            <p className="text-muted">{profile.education.degree}</p>
            <p className="mt-2 text-xs text-dim">{profile.education.dates}</p>
            <p className="mt-2 text-xs text-muted">{profile.education.notes}</p>
          </div>
          <div className="panel p-5 text-sm">
            <p className="text-xs tracking-widest text-muted uppercase">Interests</p>
            <ul className="mt-3 space-y-1 text-muted">
              {profile.interests.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        </aside>
      </div>
    </article>
  );
}
