/** Dev: Vite rewrites `/api/*` → backend. Production: set `VITE_API_URL` to full API origin (no trailing slash). */
const base = import.meta.env.VITE_API_URL ?? "/api";

export type StatusPayload = {
  api_status: string;
  lstm_model_version: string;
  mlp_model_version: string;
  data_freshness_minutes: number | null;
  capabilities: Record<string, boolean>;
  ensemble_event_model_loaded: boolean;
};

export type LightingClassScore = {
  class_id: number;
  label: string;
  probability: number;
};

export type LightingPrediction = {
  predicted_class_id: number;
  predicted_label: string;
  class_probabilities: LightingClassScore[];
  ensemble_weights: Record<string, number>;
  model_note: string;
};

export type CoachRecommendation = {
  predicted_label: string;
  shooting_mode: string;
  iso_suggestion: string;
  aperture_guidance: string;
  shutter_guidance: string;
  white_balance: string;
  gear_notes: string;
  checklist: string[];
  creative_brief: string;
  source: "rules" | "rules+openai";
  llm_addon: string | null;
};

export type PredictFromLocationResponse = {
  latitude: number;
  longitude: number;
  reference_time_utc: string;
  weather_snapshot: Record<string, number>;
  prediction: LightingPrediction;
  coach: CoachRecommendation;
};

export type EmailSubscriptionPayload = {
  email: string;
  latitude: number;
  longitude: number;
  past_hours?: number;
  enabled: boolean;
};

function tryParseJson(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

function detailToMessage(detail: unknown): string | null {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const parts: string[] = [];
    for (const item of detail) {
      if (item && typeof item === "object" && "msg" in item) {
        const o = item as { loc?: unknown[]; msg?: unknown };
        const loc = Array.isArray(o.loc)
          ? o.loc
              .filter((x): x is string | number => x !== "body" && typeof x !== "object")
              .join(".")
          : "";
        const msg = String(o.msg ?? "");
        parts.push(loc ? `${loc}: ${msg}` : msg);
      } else {
        parts.push(typeof item === "string" ? item : JSON.stringify(item));
      }
    }
    const joined = parts.filter(Boolean).join("; ");
    return joined || null;
  }
  return null;
}

/** Readable message for failed API responses (FastAPI `detail`, validation errors, short plain text). */
export function messageFromFailedResponse(res: Response, bodyText: string): string {
  const parsed = tryParseJson(bodyText);
  if (parsed && typeof parsed === "object" && parsed !== null && "detail" in parsed) {
    const fromDetail = detailToMessage((parsed as { detail: unknown }).detail);
    if (fromDetail) {
      if (res.status === 422) return `Check your input — ${fromDetail}`;
      return fromDetail;
    }
  }

  if (res.status === 502 || res.status === 503) {
    return "The service is temporarily unavailable. Try again in a moment.";
  }

  if (res.status === 504) {
    return "The request took too long. Try again with a shorter time window or later.";
  }

  if (bodyText) {
    const line = bodyText.trim().split(/\r?\n/)[0]?.trim() ?? "";
    if (line && !line.startsWith("<")) {
      return line.length > 240 ? `${line.slice(0, 237)}…` : line;
    }
  }

  if (res.status === 500) return "Something went wrong on the server.";
  return res.statusText || "Request failed";
}

async function parseJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(messageFromFailedResponse(res, text));
  }
  return res.json() as Promise<T>;
}

export async function fetchStatus(): Promise<StatusPayload> {
  const res = await fetch(`${base}/status`);
  return parseJson<StatusPayload>(res);
}

export async function fetchHealth(): Promise<{ status: string }> {
  const res = await fetch(`${base}/health`);
  return parseJson(res);
}

/** Demo payload: neutral weather; replace with real Open-Meteo + PyEphem pipeline values. */
export function demoLightingRequest() {
  const row = [40.71, -74.01, 18, 65, 10000, 40, 30, 20, 3, 15];
  const sequence = Array.from({ length: 6 }, () => [...row]);
  const tabular = [
    ...row,
    17.5,
    64,
    38,
    17.2,
    63,
    50,
    17.0,
    62,
    37,
    0.3,
    1.0,
  ];
  return { sequence, tabular };
}

export async function predictLightingEvent(body: {
  sequence: number[][];
  tabular: number[];
}): Promise<LightingPrediction> {
  const res = await fetch(`${base}/predict/event`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return parseJson<LightingPrediction>(res);
}

export async function predictFromLocation(body: {
  latitude: number;
  longitude: number;
  past_hours?: number;
}): Promise<PredictFromLocationResponse> {
  const res = await fetch(`${base}/predict/event/from_location`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return parseJson<PredictFromLocationResponse>(res);
}

/** Re-run rules (+ optional OpenAI) coach for an existing prediction without refetching weather. */
export async function coachShooting(body: {
  predicted_class_id: number;
  predicted_label: string;
  class_probabilities: LightingClassScore[];
  weather_snapshot?: Record<string, number>;
}): Promise<{ coach: CoachRecommendation }> {
  const res = await fetch(`${base}/coach/shooting`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...body,
      weather_snapshot: body.weather_snapshot ?? {},
    }),
  });
  return parseJson<{ coach: CoachRecommendation }>(res);
}

export type EmailSubscriptionResponse = {
  status: "subscribed" | "updated" | "unsubscribed";
  subscription: Record<string, unknown>;
};

export async function setEmailSubscription(body: EmailSubscriptionPayload): Promise<EmailSubscriptionResponse> {
  const res = await fetch(`${base}/notifications/email-subscription`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...body,
      past_hours: body.past_hours ?? 72,
    }),
  });
  return parseJson<EmailSubscriptionResponse>(res);
}
