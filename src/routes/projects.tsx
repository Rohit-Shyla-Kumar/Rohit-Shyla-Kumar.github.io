import { createFileRoute, Link } from "@tanstack/react-router";
import { PageHeader } from "@/components/page-header";
import { projects, type Project } from "@/content/projects";

export const Route = createFileRoute("/projects")({ component: Projects });

function Projects() {
  return (
    <div>
      <PageHeader
        title="Projects"
        lede="Bank platforms, research, and a few things built for the joy of it."
      />
      <div className="stagger-in grid gap-4 md:grid-cols-2">
        {projects.map((project) => (
          <ProjectCard key={project.id} project={project} />
        ))}
      </div>
    </div>
  );
}

function ProjectBody({ project }: { project: Project }) {
  return (
    <>
      <div className="flex items-baseline justify-between gap-3">
        <h2 className="font-display text-lg tracking-tight">{project.name}</h2>
        <span className="text-xs tabular-nums text-muted">{project.years}</span>
      </div>
      <p className="mt-1 text-xs text-dim">{project.role}</p>
      <p className="mt-3 text-sm leading-relaxed text-muted">{project.summary}</p>
      <ul className="mt-3 space-y-1 text-sm text-fg/90">
        {project.points.map((point) => (
          <li key={point} className="flex gap-2">
            <span className="mt-2 size-1 shrink-0 rounded-full bg-fg" />
            <span>{point}</span>
          </li>
        ))}
      </ul>
      <div className="mt-4 flex flex-wrap gap-2">
        {project.tags.map((tag) => (
          <span key={tag} className="rounded-md border border-line px-2 py-0.5 text-2xs text-muted">
            {tag}
          </span>
        ))}
      </div>
    </>
  );
}

function ProjectCard({ project }: { project: Project }) {
  const body = <ProjectBody project={project} />;
  const className =
    "panel block p-6 transition-colors duration-150 hover:border-fg/40";
  if (project.href?.startsWith("/blog/")) {
    const slug = project.href.replace("/blog/", "");
    return (
      <Link to="/blog/$slug" params={{ slug }} className={className}>
        {body}
      </Link>
    );
  }
  if (project.href) {
    return (
      <a href={project.href} target="_blank" rel="noreferrer" className={className}>
        {body}
      </a>
    );
  }
  return <article className={className}>{body}</article>;
}
