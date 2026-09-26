export type Project = {
  id: string;
  name: string;
  role: string;
  years: string;
  tags: string[];
  href?: string;
  summary: string;
  points: string[];
};

export const projects: Project[] = [
  {
    id: "linux-profile",
    name: "Ansible Linux Profile",
    role: "Project lead",
    years: "2024 – 2026",
    tags: ["Ansible", "RHEL", "Security", "Estate-wide"],
    summary:
      "One profile to enforce OS, filesystem, cybersecurity and tooling standards across a global bank estate — and a USD 30 million licensing save as a side effect of doing it properly.",
    points: [
      "Codified the standard instead of arguing it in tickets.",
      "CVE drift remediation and SLA impact analysis wired in.",
      "Consumed by application teams as IaaS, not as a lecture.",
    ],
  },
  {
    id: "fps-ai",
    name: "General AI for FPS",
    role: "Research & development",
    years: "2018 – 2019",
    tags: ["Reinforcement learning", "ViZDoom", "CS 1.6", "Python"],
    href: "/blog/teaching-bots-to-play",
    summary:
      "A generalist agent for classic first-person shooters. Doom (1998) and Counter-Strike 1.6, learned with deep RL, documented in public as a journal.",
    points: [
      "ViZDoom and screen-capture CS 1.6 environments.",
      "DQN / actor-critic stacks, CNN vision front-ends.",
      "Full process journal on this site.",
    ],
  },
  {
    id: "pineal",
    name: "Pineal — Aid for Visually Impaired Bowlers",
    role: "Software engineer",
    years: "2017 – 2018",
    tags: ["Computer vision", "Android", "Accessibility"],
    href: "https://github.com/Rohit-Shyla-Kumar/AidForVisualBowlers",
    summary:
      "Camera-based pin detection that tells visually impaired bowlers what is standing. Funded by the Jockey Club Youth Empathy Project; shown at World Maker Faire New York 2018.",
    points: [
      "Won Technologies for the Elderly and Disabled Makeathon.",
      "Best of the Best — Best Innovation 2017.",
      "Image processing on-device, spoken result to the bowler.",
    ],
  },
  {
    id: "one-data",
    name: "One Data Dashboard",
    role: "Project lead",
    years: "2019 – 2021",
    tags: ["Grafana", "Analytics", "Multi-region"],
    summary:
      "Enterprise dashboards that aggregate and correlate operational datasets across regions so leadership can see one picture instead of twelve exports.",
    points: [
      "Real-time compliance and platform health.",
      "Built as a graduate, still the kind of thing I would build now.",
    ],
  },
  {
    id: "wishlist",
    name: "Wishlist Invest",
    role: "Builder",
    years: "2026",
    tags: ["JavaScript", "Node", "SQLite", "Investing"],
    href: "https://github.com/Rohit-Shyla-Kumar/wishlist-invest",
    summary:
      "Buy it now, or let the money compound? A small app that prices a wishlist against future value at 5 / 10 / 25 / 50 years, with cooldowns so impulse has to wait.",
    points: [
      "Per-item FX, session auth, local SQLite.",
      "The question the dashboard asks is the real product.",
    ],
  },
  {
    id: "dsindia",
    name: "Data Science for India",
    role: "Regional director & curriculum",
    years: "2017 – 2018",
    tags: ["Education", "Curriculum", "Outreach"],
    summary:
      "UC Berkeley-based programme taking data science into Indian secondary schools and universities. I wrote curriculum and ran regional outreach.",
    points: [
      "Interdisciplinary on purpose — not a bootcamp clone.",
    ],
  },
  {
    id: "dispenser",
    name: "Minimal Medicine Dispenser",
    role: "Builder",
    years: "2017",
    tags: ["Robotics", "Hardware"],
    summary:
      "A robot that stores doses and reminds you to take them. First place, Smarter Life Aided by Robots.",
    points: ["Hardware plus software, not a slide deck."],
  },
  {
    id: "adas",
    name: "ADAS / Computer Vision",
    role: "Intern, Horus Intellisys",
    years: "2017",
    tags: ["C++", "Vision", "Debian"],
    summary:
      "India's first Advanced Driver Assistance System — classifier work and a C++ vision path taken from 1s to 4ms. Also led a McDonald's drive-through PoC and a safer-bus-stop experiment.",
    points: ["Precision +15% on the classifier.", "gdb, g++, Make, no mystery."],
  },
  {
    id: "monopoly",
    name: "Digital Monopoly (Hong Kong)",
    role: "Builder",
    years: "2018",
    tags: ["Java", "Processing"],
    href: "https://github.com/Rohit-Shyla-Kumar/DigitalMonopoly",
    summary:
      "Monopoly, but the board is Hong Kong. Course and recreation — the first kind of systems design that still feels like play.",
    points: ["Java + Processing, local rules, local streets."],
  },
];
