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
    "A journal of the final-year project: a generalist agent for Doom and Counter-Strike 1.6, written through my final year of university.",
    body: `
    ## Introduction
    Welcome to my final year project blog, "Adventures in teaching bots how to kill people". This is where I document my thoughts and milestones while working on my final year project at the School of Creative Media, City University of Hong Kong. I update this page at least once a week in the form of chapters. My final goal for this project, titled "AI For General FPS" is a bot fairly capable of playing at the level of playing multiple different FPS games at the level of an average human player.
    If you follow developments in the AI field closely, you've probably already seen the Dota 2 match between Open AI bot and Dendi. In case you haven't, check it out <a href = https://www.youtube.com/watch?v=7U4-wvhgx0w style="color:blue">here.</a></p>
    Well, if I was going to pull off anything even close to this crazy, I'd need to find out what I'm up against first, so I started going through some of the literature on the topic, this is pretty much the very least you should be familiar with if you want to do something like this yourself.</p>
                <p class="mbr-text align-left mb-0 mbr-fonts-style display-7">
                  <div>
                  <br/><a style="color:blue" href="https://www.rand.org/content/dam/rand/pubs/papers/2008/P550.pdf">The Theory of Dynammic Programming by Richard Bellman </a> <br>
                  <a style="color:blue" href="https://pdfs.semanticscholar.org/968b/ab782e52faf0f7957ca0f38b9e9078454afe.pdf"> A Survey of Applications of Markov Decision Processes by D. J. White</a><br>
                  <a style="color:blue" href="https://link.springer.com/article/10.1007/BF00115009"> Learning to predict by the methods of temporal differences by Richard S. Sutton </a><br>
                  </div>
                </p>
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
