export function PageHeader({
  kicker,
  title,
  lede,
}: {
  kicker?: string;
  title: string;
  lede?: string;
}) {
  return (
    <header className="stagger-in mb-10 max-w-2xl space-y-3">
      {kicker ? (
        <p className="text-xs tracking-widest text-muted uppercase">{kicker}</p>
      ) : null}
      <h1 className="font-display text-3xl tracking-tight text-fg">{title}</h1>
      {lede ? <p className="text-base leading-relaxed text-muted">{lede}</p> : null}
    </header>
  );
}
