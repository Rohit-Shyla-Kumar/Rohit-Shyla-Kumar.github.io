import { profile } from "@/content/profile";

function FlagIN() {
  return (
    <svg viewBox="0 0 21 15" className="size-full" aria-hidden="true">
      <rect width="21" height="5" fill="#FF9933" />
      <rect y="5" width="21" height="5" fill="#fff" />
      <rect y="10" width="21" height="5" fill="#138808" />
      <circle cx="10.5" cy="7.5" r="1.7" fill="none" stroke="#000080" strokeWidth="0.55" />
      <circle cx="10.5" cy="7.5" r="0.35" fill="#000080" />
    </svg>
  );
}

function FlagHK() {
  return (
    <svg viewBox="0 0 21 15" className="size-full" aria-hidden="true">
      <rect width="21" height="15" fill="#DE2910" />
      <g fill="#fff" transform="translate(10.5 7.5)">
        {[0, 72, 144, 216, 288].map((deg) => (
          <ellipse
            key={deg}
            rx="1.05"
            ry="2.4"
            transform={`rotate(${deg}) translate(0 -1.6)`}
          />
        ))}
        <circle r="0.7" fill="#DE2910" />
      </g>
    </svg>
  );
}

function FlagGB() {
  return (
    <svg viewBox="0 0 21 15" className="size-full" aria-hidden="true">
      <rect width="21" height="15" fill="#012169" />
      <path d="M0 0 L21 15 M21 0 L0 15" stroke="#fff" strokeWidth="3" />
      <path d="M0 0 L21 15" stroke="#C8102E" strokeWidth="1.2" />
      <path d="M21 0 L0 15" stroke="#C8102E" strokeWidth="1.2" />
      <path d="M10.5 0 V15 M0 7.5 H21" stroke="#fff" strokeWidth="5" />
      <path d="M10.5 0 V15 M0 7.5 H21" stroke="#C8102E" strokeWidth="3" />
    </svg>
  );
}

const flags = {
  IN: FlagIN,
  HK: FlagHK,
  GB: FlagGB,
} as const;

export function PlaceFlags({ className = "" }: { className?: string }) {
  return (
    <ul className={`flex items-center gap-2 ${className}`} aria-label="Places worked">
      {profile.places.map((place) => {
        const Flag = flags[place.code];
        return (
          <li key={place.code} title={place.name}>
            <span className="block h-3.5 w-5 overflow-hidden rounded-xs border border-line shadow-sm">
              <Flag />
            </span>
            <span className="sr-only">{place.name}</span>
          </li>
        );
      })}
    </ul>
  );
}
