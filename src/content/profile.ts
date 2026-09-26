export const profile = {
  name: "Rohit Shyla Kumar",
  title: "Systems Engineer",
  tagline: "Systems engineer. Builder. Writer.",
  email: "rohitshylakumar@protonmail.com",
  summary:
    "I configure and maintain the linux systems a global bank depends on - so they stay up, stay secure, and mostly run themselves. I also write about AI, engineering, investing, history and philosophy.",
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
    "Investing",
    "Chess",
    "Theatre",
    "Football",
    "Cricket",
    "Writing",
    "Politics & International relations",
  ],
  about: [
    "I am a systems engineer. For seven years at HSBC I have designed, built, and looked after Linux and cloud platforms used by retail banking, commercial banking, and securities trading - including market systems where being down is not an option.",
    "The work I care about starts with how a machine is actually built and ends with automation so good, I could 'working from home' on a beach in Sai Kung and no one notices. I've lead the modernization of a vast global Linux estate that took more than USD 30 million of licensing cost off the books.",
    "My design philosophy is to start from first principles, build, test and iterate constantly, all while optimizing every last ounce of performance out of the hardware I have. Physics is the law. Everything else is just a recommendation.",
    "Outside work I like to read history and philosophy, invest my money for outsized returns, play video games, and write. I also obsess over privacy centric, open source technology. I'm extremely opinionated on topics I understand extensively and unnaturally chill about the things I don't. This site is my public notebook.",
  ],
} as const;
