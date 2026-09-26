import { createFileRoute } from "@tanstack/react-router";
import { Github, Linkedin, Mail } from "lucide-react";
import { PageHeader } from "@/components/page-header";
import { profile } from "@/content/profile";

export const Route = createFileRoute("/contact")({ component: Contact });

const icons = {
  github: Github,
  linkedin: Linkedin,
  mail: Mail,
};

function Contact() {
  return (
    <div>
      <PageHeader
        title="Contact"
        lede="If something here is useful, wrong, or worth arguing about — write. I read mail."
      />
      <ul className="stagger-in grid gap-3 sm:grid-cols-2">
        {profile.socials.map((social) => {
          const Icon = icons[social.id as keyof typeof icons];
          return (
            <li key={social.id}>
              <a
                href={social.href}
                target={social.href.startsWith("http") ? "_blank" : undefined}
                rel={social.href.startsWith("http") ? "noreferrer" : undefined}
                className="panel flex min-h-14 items-center gap-4 px-5 transition-colors duration-150 hover:border-fg/40"
              >
                {Icon ? <Icon className="size-4" /> : <span className="text-muted">@</span>}
                <span>
                  <span className="block text-sm">{social.label}</span>
                  <span className="block text-xs break-all text-muted">
                    {social.href.replace(/^https?:\/\//, "")}
                  </span>
                </span>
              </a>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
