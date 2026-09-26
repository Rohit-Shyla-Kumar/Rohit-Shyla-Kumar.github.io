import { createFileRoute, Link } from "@tanstack/react-router";
import { PageHeader } from "@/components/page-header";
import { postsByDate } from "@/content/posts";

export const Route = createFileRoute("/blog/")({ component: BlogIndex });

function BlogIndex() {
  const posts = postsByDate();
  return (
    <div>
      <PageHeader
        title="Blog"
        lede="Notes on AI, engineering, investing, and philosophy."
      />
      <ul className="stagger-in space-y-4">
        {posts.map((post) => (
          <li key={post.slug}>
            <Link
              to="/blog/$slug"
              params={{ slug: post.slug }}
              className="panel block p-6 transition-colors duration-150 hover:border-fg/40"
            >
              <p className="text-xs tabular-nums text-muted">
                {post.date}
                <span className="ml-3 tracking-wide text-dim">
                  {post.tags.join(" · ")}
                </span>
              </p>
              <h2 className="mt-2 font-display text-xl tracking-tight">{post.title}</h2>
              <p className="mt-2 text-sm leading-relaxed text-muted">{post.excerpt}</p>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
