import type { ErrorComponentProps } from "@tanstack/react-router";
import { Link } from "@tanstack/react-router";

const FALLBACK_MESSAGE = "Something went wrong. Try reloading the page.";

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === "string" && error) return error;
  return FALLBACK_MESSAGE;
}

export function AppErrorComponent({ error }: ErrorComponentProps) {
  return (
    <main className="flex min-h-[60vh] flex-col items-start justify-center gap-4 px-2">
      <p className="text-sm text-muted">Error</p>
      <h1 className="font-display text-2xl tracking-tight">Something went wrong</h1>
      <p className="max-w-lg text-sm break-words text-muted">{errorMessage(error)}</p>
      <Link to="/" className="text-sm underline underline-offset-4 hover:text-fg">
        Back home
      </Link>
    </main>
  );
}
