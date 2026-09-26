# Rohit Shyla Kumar — personal site

Black-and-white grid site. Content is plain TypeScript in [`src/content/`](src/content/). How to edit posts, projects, and the resume: [`src/content/README.md`](src/content/README.md).

## GitHub Pages (same URL)

This is **not** a folder of finished HTML you can drop onto GitHub Pages the way the old Mobirise site was. GitHub Pages only serves finished files. This project is source code. A small GitHub Action **builds** those files and publishes them to [rohit-shyla-kumar.github.io](https://rohit-shyla-kumar.github.io/).

That build-and-publish step is the **static publish path**.

### Can I replace everything in my current repo and push?

| What you put in the repo | Works? |
|---|---|
| This **source** (`src/`, `package.json`, …) and keep “Deploy from a branch” | **No.** Visitors would see source files, not a website. |
| This source **plus** the Action in `.github/workflows/pages.yml`, and switch Pages to **GitHub Actions** | **Yes.** This is the setup below. |
| Only the **built** folder (`.output/public` after `npm run build:pages`) with “Deploy from a branch” | **Yes**, same as the old site — but you would have to rebuild locally every time you edit a post. |

### One-time setup

1. Copy this project into the `Rohit-Shyla-Kumar.github.io` repo (replace the old files; keep the repo name so the URL stays the same).
2. Repo **Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. Push to `main`. The Action installs, builds a static site, and publishes it.

After that, edit a post in `src/content/posts.ts`, commit, push. A minute later the live URL updates.

### Local commands

```bash
npm install
npm run dev          # live preview while you edit
npm run resume       # rewrite Word + text resume from src/content
npm run build:pages  # static export in .output/public
```

`npm` is only used on your machine (or in the Action). Visitors never install anything.
