import { createFileRoute, Link, notFound } from "@tanstack/react-router";
import { Markdown } from "@/lib/markdown";
import { getPost, postsByDate } from "@/content/posts";

export const Route = createFileRoute("/blog/$slug")({
  loader: ({ params }) => {
    const post = getPost(params.slug);
    if (!post) throw notFound();
    return { post };
  },
  component: BlogPost,
  notFoundComponent: () => (
    <div className="space-y-3">
      <p className="text-muted">That post is not here.</p>
      <Link to="/blog" className="text-sm underline underline-offset-4 hover:text-fg">
        All posts
      </Link>
    </div>
  ),
});

function BlogPost() {
  const { post } = Route.useLoaderData();
  const others = postsByDate().filter((p) => p.slug !== post.slug).slice(0, 3);

  return (
    <article>
      <p className="text-xs tabular-nums text-muted">
        {post.date} · {post.tags.join(" · ")}
      </p>
      <h1 className="mt-3 font-display text-3xl tracking-tight">{post.title}</h1>
      <div className="my-8 h-px bg-line" />
      <div className="max-w-prose">
        <Markdown source={post.body} />
      </div>
      <div className="mt-12 border-t border-line pt-6">
        <Link to="/blog" className="text-sm text-muted hover:text-fg">
          ← All posts
        </Link>
        <ul className="mt-6 space-y-2 text-sm">
          {others.map((p) => (
            <li key={p.slug}>
              <Link
                to="/blog/$slug"
                params={{ slug: p.slug }}
                className="text-muted hover:text-fg"
              >
                {p.title}
              </Link>
            </li>
          ))}
        </ul>
      </div>
    </article>
  );
}
