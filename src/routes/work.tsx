import { createFileRoute } from "@tanstack/react-router";
import { PageHeader } from "@/components/page-header";
import { roles } from "@/content/experience";

export const Route = createFileRoute("/work")({ component: Work });

function Work() {
  return (
    <div>
      <PageHeader title="Work" lede="Roles, in order. HSBC from graduate to Red Hat specialist." />
      <ol className="stagger-in space-y-5">
        {roles.map((role) => (
          <li key={role.id} className="panel p-6">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <h2 className="font-display text-xl tracking-tight">{role.title}</h2>
              <p className="text-sm tabular-nums text-muted">{role.dates}</p>
            </div>
            <p className="mt-1 text-sm text-muted">
              {role.org}
              <span className="text-dim"> · {role.location}</span>
              {role.current ? (
                <span className="ml-2 rounded-md border border-fg px-1.5 py-0.5 text-2xs tracking-wider text-fg">
                  Now
                </span>
              ) : null}
            </p>
            <ul className="mt-4 space-y-2 text-sm leading-relaxed text-fg/90">
              {role.bullets.map((bullet) => (
                <li key={bullet} className="flex gap-3">
                  <span className="mt-2 size-1 shrink-0 rounded-full bg-fg" />
                  <span className="text-pretty">{bullet}</span>
                </li>
              ))}
            </ul>
          </li>
        ))}
      </ol>
    </div>
  );
}
