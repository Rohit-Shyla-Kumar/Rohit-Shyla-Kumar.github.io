# Edit this site

All public wording lives in this folder. Change a file, save, and the preview updates. For GitHub Pages, commit and push — the Action rebuilds the live site.

| File | What it is |
|---|---|
| [posts.ts](posts.ts) | Blog articles (`slug`, `title`, `date`, `tags`, `excerpt`, `body`) |
| [projects.ts](projects.ts) | Project cards on `/projects` |
| [experience.ts](experience.ts) | Jobs on `/work` and the resume |
| [profile.ts](profile.ts) | Name, summary, about, education, certs, awards, links |
| [skills.ts](skills.ts) | Skill groups and the tools list |

## Resume (Word + PDF, two pages)

The on-site resume, the Word file, and the text file are generated from the same content.

1. Edit [experience.ts](experience.ts) and [profile.ts](profile.ts).
2. Run `npm run resume` to rewrite `public/Rohit-Shyla-Kumar-Resume.docx` and `.txt`.
3. PDF: open `/resume` and use **Print / save PDF**. Print CSS is sized for **two A4 pages**.

Keep bullets short. If a printout spills onto page 3, cut a bullet rather than shrinking type further.

A GitHub Pages deploy runs `npm run resume` for you, so a push is enough once the Action is enabled.
