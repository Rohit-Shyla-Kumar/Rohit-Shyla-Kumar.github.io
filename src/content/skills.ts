export type Domain = {
  id: string;
  title: string;
  blurb: string;
  skills: { name: string; level: number }[];
};

export const domains: Domain[] = [
  {
    id: "software",
    title: "Software",
    blurb:
      "Languages are tools. The work is a clear model of the system — then Python, Go, or C++ in whichever place it fits.",
    skills: [
      { name: "C / C++", level: 95 },
      { name: "Python", level: 93 },
      { name: "Systems design", level: 90 },
      { name: "SQL / Postgres / Redis", level: 80 },
      { name: "JAVA", level: 72 },
    ],
  },
  {
    id: "automation",
    title: "Automation",
    blurb:
      "Taking repetitive work off people’s plates. If a team has to click the same thing twice, it should have been a program the first time.",
    skills: [
      { name: "Ansible", level: 96 },
      { name: "CI/CD pipelines", level: 90 },
      { name: "Kubernetes / OpenShift", level: 88 },
      { name: "Secrets & certificates", level: 86 },
    ],
  },
  {
    id: "infra",
    title: "Infrastructure & reliability",
    blurb:
      "Building computers so they start the same way in every region — then keeping important systems up. Finding the real cause of a failure, not just the first alarm.",
    skills: [
      { name: "Linux", level: 98 },
      { name: "Site reliability", level: 94 },
      { name: "Incident response", level: 92 },
      { name: "Server build & provisioning", level: 90 },
      { name: "Security hardening", level: 90 },
      { name: "Monitoring & logs", level: 88 },
      { name: "Storage & filesystems", level: 84 },
      { name: "VMware to OpenShift migration", level: 82 },
    ],
  },
  {
    id: "leadership",
    title: "Leadership & impact",
    blurb:
      "Projects ship because people across regions agreed, a standard was written down, and the next team can run it without you. A licensing save north of USD 30 million. A 10-day process cut to 2.",
    skills: [
      { name: "Faster delivery", level: 90 },
      { name: "Teaching & documentation", level: 88 },
      { name: "Technical leadership", level: 86 },
      { name: "Cost reduction", level: 85 },
      { name: "Stakeholder work", level: 84 },
      { name: "Agile facilitation", level: 82 },
      { name: "Research (vision, AI)", level: 76 },
    ],
  },
];

export const stack = [
  "Linux",
  "Ansible",
  "Kubernetes",
  "OpenShift",
  "Python",
  "Go",
  "C/C++",
  "Terraform",
  "Jenkins",
  "GitLab",
  "Docker",
  "Helm",
  "Vault",
  "Kafka",
  "Postgres",
  "Redis",
  "Splunk",
  "Grafana",
  "AWS",
  "GCP",
  "Azure",
];
