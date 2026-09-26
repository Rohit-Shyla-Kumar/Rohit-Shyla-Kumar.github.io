import { createFileRoute, Link } from "@tanstack/react-router";
import { PlaceFlags } from "@/components/place-flags";
import { postsByDate } from "@/content/posts";
import { profile } from "@/content/profile";
import { projects } from "@/content/projects";
import { roles } from "@/content/experience";
import { domains } from "@/content/skills";

export const Route = createFileRoute("/")({ component: Home });

function Home() {
  const latest = postsByDate().slice(0, 3);
  const current = roles[0]!;
  const featured = projects.slice(0, 3);

  return (
    <div className="stagger-in">
      <section className="pt-10 pb-20 sm:pt-20 sm:pb-28">
        <p className="text-sm text-muted">
          {current.title} at {current.org}
        </p>
        <h1 className="mt-5 font-display text-3xl tracking-tight sm:text-5xl">
          {profile.name}
        </h1>
        <PlaceFlags className="mt-4" />
        <p className="mt-6 max-w-xl text-lg leading-relaxed text-muted">
          {profile.summary}
        </p>
        <div className="mt-9 flex flex-wrap gap-3">
          <Link
            to="/resume"
            className="inline-flex min-h-11 items-center rounded-md bg-fg px-5 text-sm text-bg transition-opacity duration-150 hover:opacity-90 active:scale-[0.96]"
          >
            Resume
          </Link>
          <Link
            to="/contact"
            className="inline-flex min-h-11 items-center rounded-md border border-line px-5 text-sm text-fg transition-colors duration-150 hover:border-fg active:scale-[0.96]"
          >
            Get in touch
          </Link>
        </div>
      </section>

      <section className="mt-8 border-t border-line pt-12">
        <div className="mb-6 flex items-end justify-between gap-4">
          <h2 className="font-display text-xl tracking-tight">Selected work</h2>
          <Link to="/projects" className="text-sm text-muted hover:text-fg">
            All projects
          </Link>
        </div>
        <ul className="grid gap-4 md:grid-cols-3">
          {featured.map((project) => (
            <li key={project.id}>
              <Link
                to="/projects"
                className="panel block h-full p-5 transition-colors duration-150 hover:border-fg/40"
              >
                <p className="text-xs text-dim">{project.years}</p>
                <h3 className="mt-2 font-display text-lg tracking-tight">{project.name}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted">{project.summary}</p>
              </Link>
            </li>
          ))}
        </ul>
      </section>

      <section className="mt-14 border-t border-line pt-12">
        <div className="mb-6 flex items-end justify-between gap-4">
          <h2 className="font-display text-xl tracking-tight">Writing</h2>
          <Link to="/blog" className="text-sm text-muted hover:text-fg">
            All posts
          </Link>
        </div>
        <ul className="divide-y divide-line border-y border-line">
          {latest.map((post) => (
            <li key={post.slug}>
              <Link
                to="/blog/$slug"
                params={{ slug: post.slug }}
                className="flex flex-col gap-1 py-5 sm:flex-row sm:items-baseline sm:justify-between sm:gap-8"
              >
                <span className="text-fg">{post.title}</span>
                <span className="shrink-0 text-sm tabular-nums text-dim">{post.date}</span>
              </Link>
            </li>
          ))}
        </ul>
      </section>

      <section className="mt-14 border-t border-line pt-12">
        <div className="mb-6 flex items-end justify-between gap-4">
          <h2 className="font-display text-xl tracking-tight">Skills</h2>
          <Link to="/skills" className="text-sm text-muted hover:text-fg">
            Full picture
          </Link>
        </div>
        <div className="flex flex-wrap gap-2">
          {domains.map((domain) => (
            <Link
              key={domain.id}
              to="/skills"
              className="rounded-md border border-line px-3 py-2 text-sm text-muted transition-colors duration-150 hover:border-fg hover:text-fg"
            >
              {domain.title}
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
