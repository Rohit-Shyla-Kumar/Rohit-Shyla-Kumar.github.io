import { Github, Linkedin } from "lucide-react";
import { profile } from "@/content/profile";

function IconX({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="currentColor" aria-hidden="true">
      <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-4.714-6.231-5.401 6.231H2.74l7.726-8.835L1.254 2.25H8.08l4.25 5.632L18.244 2.25zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
    </svg>
  );
}

const footerSocials = profile.socials.filter((s) =>
  s.id === "github" || s.id === "x" || s.id === "linkedin",
);

function Icon({ id, className }: { id: string; className?: string }) {
  if (id === "github") return <Github className={className} />;
  if (id === "linkedin") return <Linkedin className={className} />;
  return <IconX className={className} />;
}

export function SocialLinks() {
  return (
    <ul className="flex items-center gap-1">
      {footerSocials.map((social) => (
        <li key={social.id}>
          <a
            href={social.href}
            target="_blank"
            rel="noreferrer"
            aria-label={social.label}
            className="flex size-11 items-center justify-center text-muted transition-colors duration-150 hover:text-fg"
          >
            <Icon id={social.id} className="size-4" />
          </a>
        </li>
      ))}
    </ul>
  );
}
