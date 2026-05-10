import type { LightingClassScore, LightingPrediction } from "./api";

/** API label strings from ensemble (api/ensemble_bundle.CLASS_LABELS). */
export const API_CLASS_LABELS = [
  "no_event",
  "golden_hour_only",
  "dramatic_diffusion_only",
  "golden_hour_and_diffusion",
] as const;

export type ApiClassLabel = (typeof API_CLASS_LABELS)[number];

export type VerdictTone = "neutral" | "warm" | "soft" | "peak";

export type LightingVerdict = {
  headline: string;
  subhead: string;
  takeaway: string;
  tone: VerdictTone;
  /** Short line for system notifications */
  notifySummary: string;
};

const COPY: Record<ApiClassLabel, Omit<LightingVerdict, "notifySummary"> & { notifyVerb: string }> = {
  no_event: {
    headline: "Steady, everyday light",
    subhead: "No strong golden-hour or heavy diffusion signal in the recent window.",
    takeaway:
      "Conditions look ordinary for magical light—still fine for general shooting; watch solar elevation and clouds if you’re chasing glow or drama.",
    tone: "neutral",
    notifyVerb: "Conditions look steady—no strong special-light signal.",
  },
  golden_hour_only: {
    headline: "Golden-hour character",
    subhead: "Warm, low-angle light is the dominant story in this window.",
    takeaway:
      "Prioritize directional warm tones, long shadows, and subjects that benefit from rim light. Time may be limited—shoot the idea while the angle holds.",
    tone: "warm",
    notifyVerb: "Golden-hour style light is leading the forecast.",
  },
  dramatic_diffusion_only: {
    headline: "Soft, diffused sky",
    subhead: "Clouds or haze are doing the heavy lifting—not the warm low sun.",
    takeaway:
      "Think even exposures, moody skies, and detail in highlights. Great for portraits and scenes that want gentle contrast without harsh sun.",
    tone: "soft",
    notifyVerb: "Soft / diffused light is the main signal.",
  },
  golden_hour_and_diffusion: {
    headline: "Rare mix: glow + softness",
    subhead: "The model sees both golden-hour warmth and strong diffusion together.",
    takeaway:
      "If reality matches, you get forgiving contrast with color in the sky—ideal for landscapes and environmental portraits. Verify with your eyes; this combo is uncommon.",
    tone: "peak",
    notifyVerb: "Peak mix: golden glow plus diffusion—worth stepping out.",
  },
};

export function normalizeLabel(label: string): ApiClassLabel | null {
  if (API_CLASS_LABELS.includes(label as ApiClassLabel)) return label as ApiClassLabel;
  return null;
}

export function friendlyScenarioName(label: string): string {
  const n = normalizeLabel(label);
  if (!n) return label.replace(/_/g, " ");
  return {
    no_event: "No special event",
    golden_hour_only: "Golden hour",
    dramatic_diffusion_only: "Dramatic diffusion",
    golden_hour_and_diffusion: "Golden hour + diffusion",
  }[n];
}

export function confidenceTier(topProbability: number): { label: string; hint: string } {
  if (topProbability >= 0.55) return { label: "High confidence", hint: "The ensemble agrees this scenario is likely." };
  if (topProbability >= 0.3) return { label: "Moderate", hint: "Worth trusting but keep an eye on the runner-up." };
  return { label: "Uncertain", hint: "Probabilities are split—use weather + sun position to decide." };
}

export function verdictForPrediction(prediction: LightingPrediction): LightingVerdict {
  const winner =
    prediction.class_probabilities.find((c) => c.class_id === prediction.predicted_class_id) ??
    prediction.class_probabilities.reduce((a, b) => (a.probability >= b.probability ? a : b));
  const key = normalizeLabel(prediction.predicted_label) ?? normalizeLabel(winner.label);
  const base = key ? COPY[key] : null;
  const pct = Math.round(winner.probability * 100);
  if (!base) {
    return {
      headline: prediction.predicted_label.replace(/_/g, " "),
      subhead: "Model output",
      takeaway: "See probability bars below for how the ensemble splits.",
      tone: "neutral",
      notifySummary: `Lighting forecast updated (${pct}% top class).`,
    };
  }
  return {
    headline: base.headline,
    subhead: base.subhead,
    takeaway: base.takeaway,
    tone: base.tone,
    notifySummary: `${base.notifyVerb} (${pct}% ${friendlyScenarioName(winner.label)}).`,
  };
}

export function formatProbabilitiesForNotify(prediction: LightingPrediction): string {
  const sorted = [...prediction.class_probabilities].sort((a, b) => b.probability - a.probability);
  return sorted
    .slice(0, 2)
    .map((c) => `${friendlyScenarioName(c.label)} ${(c.probability * 100).toFixed(0)}%`)
    .join(" · ");
}

export function runnerUpInsight(scores: LightingClassScore[], predictedId: number): string | null {
  const others = scores.filter((s) => s.class_id !== predictedId).sort((a, b) => b.probability - a.probability);
  const second = others[0];
  if (!second || second.probability < 0.12) return null;
  return `Also watch: ${friendlyScenarioName(second.label)} at ${(second.probability * 100).toFixed(0)}%.`;
}
