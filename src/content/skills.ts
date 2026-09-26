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
      { name: "Ansible", level: 99 },
      { name: "Microservice Architecture", level: 96 },
      { name: "CI/CD pipelines", level: 90 },
      { name: "Secrets & certificates", level: 86 },
      { name: "Containerization (Kubernetes / Podman / Docker)", level: 80 },
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
      { name: "Security hardening", level: 90 },
      { name: "Monitoring & logs", level: 88 },
      { name: "Storage & filesystems", level: 84 },
    ],
  },
  {
    id: "leadership",
    title: "Leadership & impact",
    blurb:
      "Projects ship because people across regions agreed, a standard was written down, and the next team can run it without you. A licensing save north of USD 30 million. A 10-day process cut to 2.",
    skills: [
      { name: "Teaching & Communication", level: 95 },
      { name: "Technical leadership", level: 88 },
      { name: "Agile facilitation", level: 82 },
      { name: "Research", level: 79 },
      { name: "Faster delivery", level: 73 },
    ],
  },
];

export const stack = [
  "Linux",
  "Ansible",
  "Jenkins",
  "Python",
  "C/C++",
  "JAVA",
  "Kubernetes",
  "OpenShift",
  "Terraform",
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
