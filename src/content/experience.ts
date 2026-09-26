export type Role = {
  id: string;
  title: string;
  org: string;
  location: string;
  dates: string;
  current?: boolean;
  bullets: string[];
};

export const roles: Role[] = [
  {
    id: "sme",
    title: "Red Hat SME",
    org: "HSBC UK",
    location: "United Kingdom",
    dates: "Jul 2026 – Present",
    current: true,
    bullets: [
      "Integrated internal Active Directory into the Red Hat platform so users get SSO rather than another password.",
      "Drove release frequency up ~30% and tightened security monitoring with modern coding assistants in the inner loop.",
    ],
  },
  {
    id: "sre",
    title: "Site Reliability Engineer",
    org: "HSBC Hong Kong",
    location: "Hong Kong",
    dates: "Jan 2024 – Jul 2026",
    bullets: [
      "Project lead for automations across Linux, Windows, VMware and OpenShift in HSBC Internal Cloud: bare-metal build, landing accounts, SSH key rotation with HashiCorp Vault, CVE drift remediation, SSL issuance and renewal.",
      "Created the Ansible Linux Profile that enforces OS, filesystem, cybersecurity and tooling standards",
      "Shipped secret-zero plugins so application teams manage Vault secrets without a human in the loop.",
      "Offered highly available, customisable IaaS to retail banking, commercial banking, and securities trading.",
      "CI/CD for JavaScript and Python onto Kubernetes with Helm; centralised logs into Kafka for pre-incident alerting on storage, latency, and tier-0 bottlenecks.",
    ],
  },
  {
    id: "linux",
    title: "Linux Platform / DevOps Engineer",
    org: "HSBC Hong Kong",
    location: "Hong Kong",
    dates: "Aug 2021 – Jan 2024",
    bullets: [
      "Ansible pipelines and low-latency Python FastAPI services to produce RHEL layer-1 images used globally on public and private cloud",
      "Authored, tested and published maintenance packs consumed across the bank.",
      "System hardening, filesystem standards, networking; tuned TCP/IP and multicast for low-latency HFT paths.",
      "Level 2 incident response on critical applications and infrastructure.",
      "Cut Linux image prepare-and-release from 10 days to 2, with better consistency and team utilisation.",
    ],
  },
  {
    id: "grad",
    title: "Technology Graduate",
    org: "HSBC Hong Kong",
    location: "Hong Kong",
    dates: "Jul 2019 – Aug 2021",
    bullets: [
      "End-to-end OS and middleware patch management automation.",
      "System-level REST APIs for telemetry: patch compliance, build version, load balancing, core metrics — Mule, Ansible, Puppet, Java, Go.",
      "Automated VMware Tools and Visual C++ .NET upgrades on Windows.",
      "Agile lead and Scrum MC across regions; Grafana observability for senior leadership.",
      "Project lead for One Data Dashboard — multi-region operational datasets, correlated, actually used.",
    ],
  },
  {
    id: "cashk",
    title: "IT Intern",
    org: "Composers and Authors Society of Hong Kong",
    location: "Hong Kong",
    dates: "Jun 2018 – Sep 2018",
    bullets: [
      "AI systems to flag copyright violations with PyTorch and Keras.",
      "Oracle databases and PL/SQL for large-catalogue analysis.",
    ],
  },
  {
    id: "horus",
    title: "ML / Design Engineer Intern",
    org: "Horus Intellisys",
    location: "Bangalore",
    dates: "May 2017 – Sep 2017",
    bullets: [
      "Classifier tuning for India's first Advanced Driver Assistance System; precision up 15%+.",
      "Optimised C++ computer-vision paths with gdb, g++ and Make on Debian — latency from 1s to 4ms.",
    ],
  },
];
