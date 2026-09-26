export type Post = {
  slug: string;
  title: string;
  date: string;
  tags: string[];
  excerpt: string;
  body: string;
};

export const posts: Post[] = [
  {
    slug: "teaching-bots-to-play",
    title: "Adventures in teaching bots how to kill people",
    date: "2019-06-01",
    tags: ["AI", "engineering", "games"],
    excerpt:
      "A journal of the final-year project: a generalist agent for Doom and Counter-Strike 1.6, written while the thing was still catching fire in interesting ways.",
    body: `This is the public journal of my final-year project at the School of Creative Media, City University of Hong Kong. The brief I gave myself was rude: a bot fairly capable of playing *multiple* first-person shooters at the level of an average human. Not a script. Not a Doom-only trick. A generalist.

If you were paying attention around then you had already watched OpenAI's Dota agent take games off Dendi. I was not going to match that compute. I was going to find out what I was actually up against.

## The shape of the problem

An FPS agent has to see, decide, and act on a clock the environment does not pause. That is a Markov decision process whether or not you enjoy the phrase. Bellman's dynamic programming, Sutton's temporal differences, Q-learning as "the Markov equation with a max()" — the theory is small. The engineering is not.

Q-learning is intuitive. You reward particular states, you ask the agent for a policy that maximises return, you let the max operator do the greed. Deep Q-learning is the same idea with a convolutional front-end because the state is pixels, not a tidy vector.

For multiple games the architecture has to admit more than one vision stack. Image segmentation and recognition are not a shared free lunch across Doom and Counter-Strike. The policy head can be related; the eyes are not.

## Environments

Making a physical robot to press keys is comedy, not a methodology. You need an environment with direct access to controls.

ViZDoom was the honest starting point: frame buffer, game variables, a Python API, scenarios that range from "room with one enemy" to "defend the centre". OpenAI Gym sat underneath as the glue. DeepMind Lab looked tempting and then spent a season being buggy on both Windows and Linux. Counter-Strike 1.6 had no bot-friendly platform worth the name, so the path there was screen grab, a tiny key-event layer, and the acceptance that the observation is now a real desktop.

The first script always presses random keys. If that loop is wrong, no amount of DQN will save you.

## What actually moved

I trained on simple Doom scenarios first. Kills per episode went from noise to something I would not be ashamed of in a lab demo. Against me in a deathmatch it averaged about four kills a game while I managed nine — and I have been playing the game for years. It was not hugging walls. It was not a loop. That robustness is the thing I still care about more than a leaderboard screenshot.

Papers I kept on the desk: *Playing FPS Games with Deep Reinforcement Learning*, the original Atari DQN, Arnold at CMU, actor-critic curriculum work, Deep Successor RL, population-based DM work in Quake. The lesson across them is boring and true: the environment and the curriculum do more than the clever layer.

I wrote the networks in TensorFlow because I already had scars there. PyTorch was winning the paper race. Tools are not identity.

## What I would still defend

A generalist FPS agent is an AI ethics object as much as a systems one. The project title was a joke with a straight face. Teaching a machine to aim is easy to aestheticise and easier to industrialise. I wanted the notes public so the work could be argued with, not just graded.

The code from that year is the kind of code you write when you are also writing the report: \`grabscreen.py\`, \`keys.py\`, an agent that is a little too aware of the laptop fan. I would build it differently now. I would not skip the journal.

Full references from the original chapters — Bellman, Sutton, Mnih, Chaplot, and the rest — are still the right reading list if you want to follow the same corridor.
`,
  },
  {
    slug: "linux-images-in-two-days",
    title: "Linux images in two days",
    date: "2024-03-12",
    tags: ["engineering", "linux"],
    excerpt:
      "We used to spend ten days preparing and releasing a RHEL image. The interesting part is not the number. It is what the ten days were actually made of.",
    body: `A bank does not run "a Linux box". It runs a population. Different regions, different clouds, different auditors, one expectation: the next host should be boring.

When I joined the Linux platform side of HSBC the image cycle was ten days. Not because Red Hat is slow. Because the work was a relay of humans: a hardening decision in one team, a filesystem standard in another, a driver that needed a vendor email, a sign-off that lived in a queue, a test pass that started from scratch when any of those moved. Ten days was the sum of waiting, not the sum of compute.

## What we changed

The job was to make a RHEL layer-1 image that could be consumed globally on public and private cloud without each region performing folk ritual. Ansible pipelines did the ritual. Low-latency Python services did the reporting. Maintenance packs became a product with a version, not a zip in a ticket.

Once the image is a product, the rest of the estate can be opinionated. The Ansible Linux Profile — OS, filesystem, cybersecurity, tooling — is the same idea one layer up. Standards that used to be slideware became code that fails closed. The licensing save (north of USD 30 million) is what happens when you stop paying for five ways to do one job.

## The unromantic mechanics

- **Drift is the enemy, not the CVE.** A CVE is a fact. Drift is how facts become incidents. Remediation without inventory is theatre.
- **Secrets are a pipeline.** SSH keys and certificates that require a human to copy them will be copied badly. Vault, or you are lying to yourself about rotation.
- **Images are an API.** If application teams cannot provision without a conversation, you do not have a platform. You have a guild.

We took the cycle from ten days to two. Consistency went up because the path got narrower, not because people typed faster. Resource utilisation went up because skilled engineers stopped being glue.

## What two days still means

Two days is not CI for a container. It is still an operating system with firmware, filesystems, and a compliance story. Anyone who tells you "just use a distroless image" has not sat with a trading rack, a regulator, and a kernel module that has a name.

The right ambition is not zero days. It is a cycle whose steps you can name, time, and replay. If you cannot replay it, you do not own it.

I still think of platform engineering as a kindness to the future on-call. The image that boots the same way is a letter to that person.
`,
  },
  {
    slug: "first-principles-against-folklore",
    title: "First principles against folklore",
    date: "2025-11-02",
    tags: ["philosophy", "engineering"],
    excerpt:
      "Production fills with sentences that sound like wisdom and behave like superstition. First principles is not a brand. It is a refusal to inherit a step you cannot justify.",
    body: `Every estate accumulates folklore. Restart it twice. Don't touch that cron. We always set this sysctl. The ticket that fixed it last time. None of this is stupid. Folklore is compressed incident memory. It is also how a platform becomes a museum.

First principles, as I use the phrase at work, is not Descartes with a lanyard. It is a habit: if I cannot point at the mechanism, I do not get to treat the rule as true.

## Three questions

1. **What is the invariant?** Not the tool. The invariant. "Keys must exist only in Vault and live for less than N days" is an invariant. "Use this Ansible role" is a tactic.
2. **What fails, and how do we see it?** A control without an observation is a wish. Logging that cannot answer "which host drifted from baseline at 02:13" is décor.
3. **Who is the next operator?** If the answer is "me, because I remember", the system is unfinished.

These questions are philosophical in the old sense: they are about what we claim to know. Socrates is not a stretch here. Most outages I have worked were not novel physics. They were a story we had agreed not to re-examine.

## Ethics is not a slide at the end

I used to write "AI ethicist" on a student website because I had been near models that classify people and scenes, and I did not like the shrug. The shrug has a production form. We ship automation that can take a host out of a trading path, rotate every secret, or deny a build. That is moral work even when the payload is YAML.

A good automation is legible. A bad one is a ritual with a logo. Legibility is how you stay accountable to the people who did not attend the design review — including the ones who will be asleep when it fires.

## What I keep from the old games

The FPS agent taught me a cheap lesson that survived contact with a bank: the environment is the curriculum. If your staging is a cartoon of production, you will train a cartoon policy. If your pipeline tests the image the traders actually boot, you will learn the real failure modes.

Philosophy without an environment is a blog. Engineering without philosophy is folklore with better CI.

I would rather be slow on a sentence and fast on a rollback.
`,
  },
  {
    slug: "buy-it-or-buy-the-future",
    title: "Buy it, or buy the future?",
    date: "2026-05-22",
    tags: ["investing", "engineering"],
    excerpt:
      "A wishlist is a list of future selves. I built a small app that makes the price of impatience visible, then I had to live with the chart.",
    body: `Most of my professional life is latency, availability, and the cost of a mistake on a host. Investing is the same shape with worse feedback. The loss does not page you. It just sits there, compounding in the wrong direction, while you buy headphones.

I wrote a small tool, [wishlist-invest](https://github.com/Rohit-Shyla-Kumar/wishlist-invest), because I was tired of doing the arithmetic in my head and then doing the purchase anyway. You put the object in: name, price, currency, a cooldown, an assumed annual return. It tells you what that money becomes at 5, 10, 25, 50 years. Then it makes you wait.

## The formula is not the point

Future value \`= price × (1 + r)^n\` is something a spreadsheet already knows. The product is the pause. A cooldown is an admission that System 1 should not be allowed to SSH into the bank account.

I used to think discipline was a personality trait. It is closer to a platform problem. If the default path is "click buy", you will buy. If the default path is "sit with the 25-year number", you still might buy, but you will know the trade.

## What production taught me about money

- **Toil has a net present value.** Every manual certificate renewal is a tiny negative yield. Automate it and you have bought back hours that compound.
- **Single points of failure are concentration risk.** A career, a stock, a region, a cloud. The slide is always "we are diversified". The graph is usually not.
- **Observability first.** I will not hold an opinion about a portfolio I do not measure. Grafana for hosts; a ledger for cash. Same instinct.

I am not a financial adviser. I am a person who likes machines that tell the truth on a schedule. The app stores items in SQLite, converts with illustrative FX, and refuses to pretend a cooldown is a vibe.

If you want the object, buy the object. Just do not tell yourself it was free. The future value is the receipt you did not print.
`,
  },
];

export function getPost(slug: string) {
  return posts.find((post) => post.slug === slug);
}

export function postsByDate() {
  return [...posts].sort((a, b) => (a.date < b.date ? 1 : -1));
}
