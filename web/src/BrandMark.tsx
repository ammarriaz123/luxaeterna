type Props = {
  size?: number;
  className?: string;
  variant?: "full" | "minimal";
};

/**
 * LuxAeterna mark: a circular dial framing a snow-capped horizon with a low sun.
 * Drawn at viewBox 80×80 so hairlines stay crisp at any size.
 */
export function BrandMark({ size = 56, className, variant = "full" }: Props) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 80 80"
      fill="none"
      className={className}
      aria-hidden
      role="img"
    >
      <defs>
        <linearGradient id="lx-sky" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#efe1f4" />
          <stop offset="100%" stopColor="#fbe7ec" />
        </linearGradient>
        <linearGradient id="lx-mtn-far" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#c9a9e0" />
          <stop offset="100%" stopColor="#a47cc4" />
        </linearGradient>
        <linearGradient id="lx-mtn-near" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#8b65b0" />
          <stop offset="100%" stopColor="#5d3f7c" />
        </linearGradient>
        <radialGradient id="lx-sun" cx="0.5" cy="0.5" r="0.5">
          <stop offset="0%" stopColor="#f7d39a" />
          <stop offset="100%" stopColor="#e89a7a" />
        </radialGradient>
        <clipPath id="lx-clip">
          <circle cx="40" cy="40" r="34" />
        </clipPath>
      </defs>

      {/* Outer dial ring (hairline like a watch index) */}
      <circle cx="40" cy="40" r="38" stroke="#c0a4d4" strokeWidth="0.6" opacity="0.55" />
      <circle cx="40" cy="40" r="34.5" stroke="#5d3f7c" strokeOpacity="0.45" strokeWidth="0.5" />

      {/* Sky and scene clipped to dial */}
      <g clipPath="url(#lx-clip)">
        <rect x="0" y="0" width="80" height="80" fill="url(#lx-sky)" />
        {/* Sun (low, golden hour) */}
        <circle cx="52" cy="34" r="6" fill="url(#lx-sun)" opacity="0.95" />
        {/* Faraway range */}
        <path d="M0 50 L12 38 L20 44 L30 32 L40 42 L50 30 L62 44 L74 36 L80 46 L80 80 L0 80 Z" fill="url(#lx-mtn-far)" opacity="0.7" />
        {/* Snow caps on far range */}
        <path d="M28 34 L30 32 L33 34 L34 33 L36 36 Z" fill="#fffafd" opacity="0.85" />
        <path d="M48 32 L50 30 L52 32 L54 31 L56 35 Z" fill="#fffafd" opacity="0.85" />
        {/* Closer ridge */}
        <path d="M0 58 L10 50 L20 56 L34 46 L46 56 L58 48 L70 56 L80 52 L80 80 L0 80 Z" fill="url(#lx-mtn-near)" opacity="0.95" />
        {/* Lake reflection hairline */}
        <line x1="6" y1="62" x2="74" y2="62" stroke="#fffafd" strokeOpacity="0.55" strokeWidth="0.5" />
      </g>

      {/* Inner dial border on top of scene */}
      <circle cx="40" cy="40" r="34" stroke="#5d3f7c" strokeOpacity="0.5" strokeWidth="0.8" />

      {variant === "full" && (
        <>
          {/* 12-3-6-9 hairline indices (watch face) */}
          <line x1="40" y1="3" x2="40" y2="6" stroke="#5d3f7c" strokeOpacity="0.7" strokeWidth="0.7" />
          <line x1="77" y1="40" x2="74" y2="40" stroke="#5d3f7c" strokeOpacity="0.5" strokeWidth="0.6" />
          <line x1="40" y1="77" x2="40" y2="74" stroke="#5d3f7c" strokeOpacity="0.5" strokeWidth="0.6" />
          <line x1="3" y1="40" x2="6" y2="40" stroke="#5d3f7c" strokeOpacity="0.5" strokeWidth="0.6" />
        </>
      )}

      {/* Single sakura petal seal */}
      <g transform="translate(63 63)">
        <path
          d="M0 -3.2 C 1.8 -3.2 3.0 -1.8 2.6 0 C 1.6 1.0 0.4 1.2 0 3.2 C -0.4 1.2 -1.6 1.0 -2.6 0 C -3.0 -1.8 -1.8 -3.2 0 -3.2 Z"
          fill="#efb8c8"
          stroke="#d97c95"
          strokeWidth="0.4"
        />
      </g>
    </svg>
  );
}

export function MountainBackdrop({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 1440 480"
      preserveAspectRatio="xMidYEnd slice"
      className={className}
      aria-hidden
      role="img"
    >
      <defs>
        <linearGradient id="lx-bg-far" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#d8c2ed" stopOpacity="0.55" />
          <stop offset="100%" stopColor="#b88dd9" stopOpacity="0.4" />
        </linearGradient>
        <linearGradient id="lx-bg-mid" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#b88dd9" stopOpacity="0.55" />
          <stop offset="100%" stopColor="#9a6cc1" stopOpacity="0.5" />
        </linearGradient>
        <linearGradient id="lx-bg-near" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#7a4f9e" stopOpacity="0.6" />
          <stop offset="100%" stopColor="#5d3f7c" stopOpacity="0.65" />
        </linearGradient>
      </defs>
      {/* Far range */}
      <path
        d="M0 320 L120 240 L200 280 L320 200 L420 260 L540 180 L660 240 L780 200 L900 250 L1020 210 L1160 260 L1280 220 L1440 270 L1440 480 L0 480 Z"
        fill="url(#lx-bg-far)"
      />
      {/* Snow lines on far peaks */}
      <path
        d="M310 204 L320 200 L340 215 M530 184 L540 180 L555 196 M775 204 L780 200 L800 220 M1015 214 L1020 210 L1040 225"
        stroke="#fffafd"
        strokeOpacity="0.5"
        strokeWidth="2"
        fill="none"
        strokeLinecap="round"
      />
      {/* Mid range */}
      <path
        d="M0 380 L100 320 L220 360 L360 290 L480 350 L620 300 L760 360 L880 310 L1020 360 L1180 320 L1320 360 L1440 340 L1440 480 L0 480 Z"
        fill="url(#lx-bg-mid)"
      />
      {/* Near range */}
      <path
        d="M0 430 L160 390 L320 420 L500 380 L680 420 L860 390 L1040 425 L1220 395 L1440 425 L1440 480 L0 480 Z"
        fill="url(#lx-bg-near)"
      />
    </svg>
  );
}
