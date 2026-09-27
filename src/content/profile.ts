export const profile = {
  name: "Rohit Shyla Kumar",
  title: "Systems Engineer",
  tagline: "Systems engineer. Builder. Writer.",
  email: "rohitshylakumar@protonmail.com",
  summary:
    "I design and run the computer systems a global bank depends on — so they stay up, stay secure, and mostly run themselves. I also write about AI, engineering, investing, and philosophy.",
  motd: "Everything, Everywhere, All The Time.",
  places: [
    { code: "IN", name: "India" },
    { code: "HK", name: "Hong Kong" },
    { code: "GB", name: "United Kingdom" },
  ],
  socials: [
    {
      id: "github",
      label: "GitHub",
      href: "https://github.com/Rohit-Shyla-Kumar",
    },
    {
      id: "linkedin",
      label: "LinkedIn",
      href: "https://hk.linkedin.com/in/rohit-shyla-kumar",
    },
    {
      id: "x",
      label: "X",
      href: "https://x.com/0thrayofthesun",
    },
    {
      id: "mail",
      label: "Email",
      href: "mailto:rohitshylakumar@protonmail.com",
    },
  ],
  education: {
    school: "City University of Hong Kong",
    degree: "Bachelor of Science in Creative Media",
    dates: "Aug 2016 – Jun 2019",
    notes: "CGPA 3.71 · Dean's List of Honor · Competitive Programming · English Debate",
  },
  certifications: [
    {
      name: "Red Hat Certified Engineer (RHCE)",
      detail: "Automation with Ansible",
      date: "May 2023",
    },
    {
      name: "Red Hat Certified System Administrator (RHCSA)",
      detail: "Linux system administration",
      date: "Mar 2023",
    },
    {
      name: "OpenHack: DevOps for Data Science",
      detail: "Microsoft OpenHack",
      date: "Jul 2020",
    },
  ],
  awards: [
    "Certified Enterprise Engineer — Role Model, technical & leadership programme (2023)",
    "Hong Kong SAR Government Scholarship (2018–19)",
    "Hong Kong SAR Talent Development Scholarship (2017)",
    "Smarter Life Aided by Robots — 1st place, medicine dispenser (2017)",
  ],
  interests: [
    "Philosophy",
    "Ancient cultures",
    "Chess",
    "Investing",
    "Theatre",
    "Football",
    "Cricket",
    "Writing",
    "International relations",
  ],
  about: [
    "I am a systems engineer. For six years at HSBC I have designed, built, and looked after Linux and cloud platforms used by retail banking, commercial banking, and securities trading — including market systems where being down is not an option.",
    "The work I care about starts with how a machine is actually built and ends with something a team can run without me. I write automation so people do not have to repeat the same failure. One standard I led now applies across the bank’s Linux estate and took more than USD 30 million of licensing cost off the books.",
    "Before the bank I studied creative media and could not leave computer vision or game AI alone: an internship on India’s first driver-assistance system, a camera app for visually impaired bowlers, an agent that learned to play Doom and Counter-Strike. That habit of starting from first principles never left.",
    "Outside work I read philosophy, follow markets, play chess, and write. This site is the public notebook.",
  ],
} as const;
