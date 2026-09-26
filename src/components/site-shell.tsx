import { Link } from "@tanstack/react-router";
import type { ReactNode } from "react";
import { profile } from "@/content/profile";
import { GridFloor } from "@/components/grid-floor";
import { SiteNav } from "@/components/site-nav";
import { SocialLinks } from "@/components/social-icons";

export function SiteShell({ children }: { children: ReactNode }) {
  return (
    <div className="relative min-h-dvh bg-bg text-fg">
      <GridFloor />
      <div className="relative z-10">
        <header className="no-print sticky top-0 z-20 border-b border-line/80 bg-bg/75 backdrop-blur-md">
          <div className="relative mx-auto flex max-w-5xl items-center justify-between gap-4 px-4 sm:px-6">
            <Link
              to="/"
              className="flex min-h-14 items-center font-display text-sm tracking-wide"
            >
              {profile.name}
            </Link>
            <SiteNav />
          </div>
        </header>
        <main className="mx-auto max-w-5xl px-4 py-12 sm:px-6 sm:py-16 print:max-w-none print:p-0">
          {children}
        </main>
        <footer className="no-print mx-auto max-w-5xl border-t border-line px-4 py-8 sm:px-6">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <p className="text-sm text-muted">{profile.name}</p>
            <SocialLinks />
          </div>
        </footer>
      </div>
    </div>
  );
}
