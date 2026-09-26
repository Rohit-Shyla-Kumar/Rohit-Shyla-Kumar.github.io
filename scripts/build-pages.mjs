import { spawn } from "node:child_process";
import { copyFileSync, existsSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const pub = join(root, ".output", "public");

process.env.NITRO_PRESET = "github_pages";

function finalize() {
  const index = join(pub, "index.html");
  if (!existsSync(index)) {
    console.error("[build:pages] missing .output/public/index.html");
    process.exit(1);
  }
  copyFileSync(index, join(pub, "404.html"));
  writeFileSync(join(pub, ".nojekyll"), "");
  rmSync(join(pub, "__grok"), { recursive: true, force: true });
  console.log("[build:pages] static site ready in .output/public");
}

const child = spawn(
  process.execPath,
  ["scripts/with-app-env.mjs", "vite", "build"],
  { cwd: root, stdio: "inherit", env: process.env },
);

child.on("exit", (code) => {
  if (!existsSync(join(pub, "index.html"))) {
    process.exit(code || 1);
  }
  if (code !== 0) {
    console.log(
      "[build:pages] Nitro server bundle skipped (GitHub Pages is static).",
    );
  }
  finalize();
  process.exit(0);
});
