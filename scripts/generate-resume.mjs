import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  AlignmentType,
  BorderStyle,
  Document,
  HeadingLevel,
  Packer,
  Paragraph,
  TabStopType,
  TextRun,
} from "docx";
import { roles } from "../src/content/experience.ts";
import { profile } from "../src/content/profile.ts";
import { projects } from "../src/content/projects.ts";
import { stack } from "../src/content/skills.ts";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const outDir = join(root, "public");
mkdirSync(outDir, { recursive: true });

const hairline = {
  bottom: { color: "999999", space: 1, style: BorderStyle.SINGLE, size: 6 },
};

function heading(text) {
  return new Paragraph({
    spacing: { before: 140, after: 40 },
    border: hairline,
    children: [
      new TextRun({
        text: text.toUpperCase(),
        size: 18,
        font: "Calibri",
        bold: true,
        characterSpacing: 80,
      }),
    ],
  });
}

function body(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 40 },
    children: [
      new TextRun({
        text,
        size: 20,
        font: "Calibri",
        ...opts,
      }),
    ],
  });
}

function bullet(text) {
  return new Paragraph({
    spacing: { after: 20 },
    indent: { left: 180, hanging: 140 },
    children: [
      new TextRun({ text: "•  ", size: 20, font: "Calibri" }),
      new TextRun({ text, size: 20, font: "Calibri" }),
    ],
  });
}

const roleBlocks = roles.flatMap((role) => [
  new Paragraph({
    spacing: { before: 80, after: 20 },
    tabStops: [{ type: TabStopType.RIGHT, position: 10080 }],
    children: [
      new TextRun({
        text: `${role.title} · ${role.org}`,
        size: 21,
        font: "Calibri",
        bold: true,
      }),
      new TextRun({ text: `\t${role.dates}`, size: 20, font: "Calibri", color: "555555" }),
    ],
  }),
  ...role.bullets.map((item) => bullet(item)),
]);

const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: "Calibri", size: 20 },
      },
    },
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 680, bottom: 680, left: 720, right: 720 },
        },
      },
      children: [
        new Paragraph({
          spacing: { after: 20 },
          children: [
            new TextRun({
              text: profile.name,
              size: 36,
              font: "Calibri",
              bold: true,
            }),
          ],
        }),
        new Paragraph({
          spacing: { after: 20 },
          children: [
            new TextRun({
              text: profile.tagline,
              size: 20,
              font: "Calibri",
              color: "444444",
            }),
          ],
        }),
        new Paragraph({
          spacing: { after: 80 },
          children: [
            new TextRun({
              text: `${profile.email}  ·  github.com/Rohit-Shyla-Kumar  ·  hk.linkedin.com/in/rohit-shyla-kumar`,
              size: 18,
              font: "Calibri",
              color: "555555",
            }),
          ],
        }),
        heading("Summary"),
        body(profile.summary),
        heading("Experience"),
        ...roleBlocks,
        heading("Skills"),
        body(stack.join(" · ")),
        heading("Education"),
        body(
          `${profile.education.degree}, ${profile.education.school} (${profile.education.dates})`,
          { bold: true },
        ),
        body(profile.education.notes),
        heading("Certifications"),
        ...profile.certifications.map((cert) =>
          body(`${cert.name} — ${cert.detail} (${cert.date})`),
        ),
        heading("Awards"),
        ...profile.awards.map((award) => bullet(award)),
        heading("Selected projects"),
        ...projects.slice(0, 4).map((project) =>
          body(`${project.name} (${project.years}) — ${project.summary}`),
        ),
      ],
    },
  ],
});

const docxPath = join(outDir, "Rohit-Shyla-Kumar-Resume.docx");
const txtPath = join(outDir, "Rohit-Shyla-Kumar-Resume.txt");

const txt = [
  profile.name.toUpperCase(),
  profile.tagline,
  profile.email,
  "https://github.com/Rohit-Shyla-Kumar",
  "https://hk.linkedin.com/in/rohit-shyla-kumar",
  "https://rohit-shyla-kumar.github.io/",
  "",
  "SUMMARY",
  profile.summary,
  "",
  "EXPERIENCE",
  "",
  ...roles.flatMap((role) => [
    `${role.title} · ${role.org} · ${role.dates}`,
    ...role.bullets.map((item) => `- ${item}`),
    "",
  ]),
  "SKILLS",
  stack.join(", "),
  "",
  "EDUCATION",
  `${profile.education.degree}, ${profile.education.school} (${profile.education.dates})`,
  profile.education.notes,
  "",
  "CERTIFICATIONS",
  ...profile.certifications.map((c) => `- ${c.name} — ${c.detail} (${c.date})`),
  "",
  "AWARDS",
  ...profile.awards.map((a) => `- ${a}`),
  "",
  "PROJECTS",
  ...projects.map((p) => `- ${p.name} (${p.years})`),
  "",
].join("\n");

const buffer = await Packer.toBuffer(doc);
writeFileSync(docxPath, buffer);
writeFileSync(txtPath, txt);
console.log(`wrote ${docxPath}`);
console.log(`wrote ${txtPath}`);
