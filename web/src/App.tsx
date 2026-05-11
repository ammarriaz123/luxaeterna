import { useCallback, useEffect, useRef, useState } from "react";
import {
  coachShooting,
  demoLightingRequest,
  fetchHealth,
  fetchStatus,
  predictFromLocation,
  predictLightingEvent,
  setEmailSubscription,
  type CoachRecommendation,
  type LightingPrediction,
  type PredictFromLocationResponse,
  type StatusPayload,
} from "./api";
import { BrandMark, MountainBackdrop } from "./BrandMark";
import { formatCoord, getCurrentPosition, GeolocationError, isGeolocationSupported } from "./geolocation";
import {
  confidenceTier,
  formatProbabilitiesForNotify,
  friendlyScenarioName,
  runnerUpInsight,
  verdictForPrediction,
} from "./lightingCopy";
import {
  getHourlyLastReference,
  getHourlyNotifyLocation,
  getNotifyPreference,
  isNotificationApiAvailable,
  notificationPermission,
  notifyResult,
  requestNotificationPermission,
  setHourlyLastReference,
  setHourlyNotifyLocation,
  setNotifyPreference,
} from "./notify";

function CapabilityPill({ label, on }: { label: string; on: boolean }) {
  return (
    <span className={`cap ${on ? "cap--on" : "cap--off"}`}>
      <span className="cap__dot" aria-hidden />
      {label}
    </span>
  );
}

function VerdictCard({ prediction }: { prediction: LightingPrediction }) {
  const verdict = verdictForPrediction(prediction);
  const winner = prediction.class_probabilities.find((c) => c.class_id === prediction.predicted_class_id);
  const topP = winner?.probability ?? 0;
  const conf = confidenceTier(topP);
  const runner = runnerUpInsight(prediction.class_probabilities, prediction.predicted_class_id);

  return (
    <div className={`verdict verdict--${verdict.tone}`}>
      <div className="verdict__top">
        <p className="verdict__eyebrow">What this means for you</p>
        <h3 className="verdict__headline display">{verdict.headline}</h3>
        <p className="verdict__sub">{verdict.subhead}</p>
      </div>
      <p className="verdict__takeaway">{verdict.takeaway}</p>
      <div className="verdict__meta">
        <div className="verdict__confidence">
          <span className="verdict__conf-label">{conf.label}</span>
          <span className="verdict__conf-hint muted small">{conf.hint}</span>
          <div className="verdict__conf-bar" aria-hidden>
            <div className="verdict__conf-fill" style={{ width: `${Math.min(100, topP * 100)}%` }} />
          </div>
          <span className="verdict__conf-pct mono small">{(topP * 100).toFixed(1)}% on {friendlyScenarioName(prediction.predicted_label)}</span>
        </div>
        {runner && <p className="verdict__runner small muted">{runner}</p>}
      </div>
    </div>
  );
}

function PredictionBars({ prediction }: { prediction: LightingPrediction }) {
  return (
    <div className="pred">
      <p className="pred__section-title">Ensemble breakdown</p>
      <p className="pred__section-hint small muted">How the model splits the last weather window (XGB + LSTM + MLP blend).</p>
      <ul className="pred__bars">
        {prediction.class_probabilities.map((c) => (
          <li key={c.class_id}>
            <span className="pred__label">{friendlyScenarioName(c.label)}</span>
            <div className="pred__track">
              <div
                className={`pred__fill ${c.class_id === prediction.predicted_class_id ? "pred__fill--winner" : ""}`}
                style={{ width: `${Math.min(100, c.probability * 100)}%` }}
              />
            </div>
            <span className="pred__pct mono">{(c.probability * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
      <p className="mono small muted pred__weights">
        Model blend: {(prediction.ensemble_weights.xgb * 100).toFixed(0)}% XGB · {(prediction.ensemble_weights.lstm * 100).toFixed(0)}% LSTM ·{" "}
        {(prediction.ensemble_weights.mlp * 100).toFixed(0)}% MLP
      </p>
    </div>
  );
}

function CoachPanel({
  coach,
  onRefreshCoach,
  coachRefreshing,
}: {
  coach: CoachRecommendation;
  onRefreshCoach?: () => void;
  coachRefreshing?: boolean;
}) {
  return (
    <div className="coach">
      <div className="coach__head">
        <h3 className="coach__title">Camera & shooting coach</h3>
        <span className={`coach__src coach__src--${coach.source === "rules+openai" ? "llm" : "rules"}`}>
          {coach.source === "rules+openai" ? "Rules + AI" : "Rules"}
        </span>
        {onRefreshCoach && (
          <button type="button" className="btn btn--ghost btn--sm" onClick={onRefreshCoach} disabled={coachRefreshing}>
            {coachRefreshing ? "Refreshing…" : "Refresh tips"}
          </button>
        )}
      </div>
      <p className="coach__mode">
        <span className="muted">Suggested mode</span> — {coach.shooting_mode}
      </p>
      <dl className="coach__grid">
        <div>
          <dt>ISO</dt>
          <dd>{coach.iso_suggestion}</dd>
        </div>
        <div>
          <dt>Aperture</dt>
          <dd>{coach.aperture_guidance}</dd>
        </div>
        <div>
          <dt>Shutter</dt>
          <dd>{coach.shutter_guidance}</dd>
        </div>
        <div>
          <dt>White balance</dt>
          <dd>{coach.white_balance}</dd>
        </div>
      </dl>
      <p className="coach__gear">
        <span className="muted">Gear</span> — {coach.gear_notes}
      </p>
      <div className="coach__brief">
        <h4>Creative brief</h4>
        <p>{coach.creative_brief}</p>
      </div>
      <div className="coach__check">
        <h4>Field checklist</h4>
        <ul>
          {coach.checklist.map((item, i) => (
            <li key={i}>{item}</li>
          ))}
        </ul>
      </div>
      {coach.llm_addon && (
        <div className="coach__addon">
          <h4>AI add-on</h4>
          <p className="small">{coach.llm_addon}</p>
        </div>
      )}
    </div>
  );
}

function firePredictionNotification(prediction: LightingPrediction, context: string) {
  const v = verdictForPrediction(prediction);
  const detail = formatProbabilitiesForNotify(prediction);
  notifyResult(context, `${v.notifySummary}\n${detail}`);
}

const EMAIL_NOTIFY_KEY = "luxaeterna_notify_email_on";
const EMAIL_NOTIFY_VALUE_KEY = "luxaeterna_notify_email_value";

function getEmailNotifyPreference(): boolean {
  try {
    return localStorage.getItem(EMAIL_NOTIFY_KEY) === "1";
  } catch {
    return false;
  }
}

function setEmailNotifyPreference(on: boolean): void {
  try {
    if (on) localStorage.setItem(EMAIL_NOTIFY_KEY, "1");
    else localStorage.removeItem(EMAIL_NOTIFY_KEY);
  } catch {
    // ignore
  }
}

function getSavedEmail(): string {
  try {
    return localStorage.getItem(EMAIL_NOTIFY_VALUE_KEY) ?? "";
  } catch {
    return "";
  }
}

function setSavedEmail(value: string): void {
  try {
    if (value) localStorage.setItem(EMAIL_NOTIFY_VALUE_KEY, value);
    else localStorage.removeItem(EMAIL_NOTIFY_VALUE_KEY);
  } catch {
    // ignore
  }
}

export default function App() {
  const [status, setStatus] = useState<StatusPayload | null>(null);
  const [health, setHealth] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [predLoading, setPredLoading] = useState(false);
  const [prediction, setPrediction] = useState<LightingPrediction | null>(null);
  const [predErr, setPredErr] = useState<string | null>(null);

  const [lat, setLat] = useState("40.7128");
  const [lon, setLon] = useState("-74.0060");
  const [pastHours, setPastHours] = useState("72");
  const [locLoading, setLocLoading] = useState(false);
  const [locErr, setLocErr] = useState<string | null>(null);
  const [locationBundle, setLocationBundle] = useState<PredictFromLocationResponse | null>(null);
  const [coachRefreshing, setCoachRefreshing] = useState(false);
  const [geoLoading, setGeoLoading] = useState(false);
  const [geoErr, setGeoErr] = useState<string | null>(null);

  const [notifyOn, setNotifyOn] = useState(getNotifyPreference);
  const [notifPerm, setNotifPerm] = useState<NotificationPermission | "unsupported">(
    isNotificationApiAvailable() ? notificationPermission() : "unsupported",
  );
  const [emailNotifyOn, setEmailNotifyOn] = useState(getEmailNotifyPreference);
  const [emailAddress, setEmailAddress] = useState(getSavedEmail);
  const [emailBusy, setEmailBusy] = useState(false);
  const [emailSuccess, setEmailSuccess] = useState<string | null>(null);
  const locationResultsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!emailSuccess) return;
    const t = window.setTimeout(() => setEmailSuccess(null), 5500);
    return () => window.clearTimeout(t);
  }, [emailSuccess]);

  const refresh = useCallback(async () => {
    setErr(null);
    setLoading(true);
    try {
      const [s, h] = await Promise.all([fetchStatus(), fetchHealth()]);
      setStatus(s);
      setHealth(h.status);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed to reach API");
      setStatus(null);
      setHealth(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!isNotificationApiAvailable()) return;
    setNotifPerm(Notification.permission);
    if (getNotifyPreference() && Notification.permission !== "granted") {
      setNotifyOn(false);
      setNotifyPreference(false);
    }
  }, []);

  const parseCurrentLocation = () => {
    const latitude = Number(lat);
    const longitude = Number(lon);
    const ph = Number(pastHours);
    if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return null;
    if (!Number.isFinite(ph) || ph < 24 || ph > 240) return null;
    return { latitude, longitude, pastHours: Math.round(ph) };
  };

  const runDemoPredict = async () => {
    setPredErr(null);
    setPredLoading(true);
    setPrediction(null);
    try {
      const body = demoLightingRequest();
      const p = await predictLightingEvent(body);
      setPrediction(p);
      firePredictionNotification(p, "Demo analysis ready");
    } catch (e) {
      setPredErr(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setPredLoading(false);
    }
  };

  const runLocationPredictWithValues = async (latitude: number, longitude: number, ph: number) => {
    setLocErr(null);
    setLocLoading(true);
    setLocationBundle(null);
    try {
      const bundle = await predictFromLocation({ latitude, longitude, past_hours: ph });
      setLocationBundle(bundle);
      firePredictionNotification(bundle.prediction, "Location analysis ready");
      setHourlyLastReference(bundle.reference_time_utc);
      window.setTimeout(() => {
        locationResultsRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }, 120);
    } catch (e) {
      setLocErr(e instanceof Error ? e.message : "Location prediction failed");
    } finally {
      setLocLoading(false);
    }
  };

  const runLocationPredict = async () => {
    const latitude = Number(lat);
    const longitude = Number(lon);
    const ph = Number(pastHours);
    if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
      setLocErr("Enter numeric latitude and longitude.");
      return;
    }
    if (!Number.isFinite(ph) || ph < 24 || ph > 240) {
      setLocErr("Past hours must be between 24 and 240.");
      return;
    }
    await runLocationPredictWithValues(latitude, longitude, Math.round(ph));
  };

  const useMyLocationAndAnalyze = async () => {
    setGeoErr(null);
    setGeoLoading(true);
    try {
      const pos = await getCurrentPosition();
      const latitude = pos.coords.latitude;
      const longitude = pos.coords.longitude;
      setLat(formatCoord(latitude, 5));
      setLon(formatCoord(longitude, 5));
      const ph = Number(pastHours);
      if (!Number.isFinite(ph) || ph < 24 || ph > 240) {
        setGeoErr("Set past hours between 24 and 240 before analyzing.");
        setGeoLoading(false);
        return;
      }
      setGeoLoading(false);
      await runLocationPredictWithValues(latitude, longitude, Math.round(ph));
    } catch (e) {
      const msg = e instanceof GeolocationError ? e.message : e instanceof Error ? e.message : "Location failed";
      setGeoErr(msg);
    } finally {
      setGeoLoading(false);
    }
  };

  const refreshCoachOnly = async () => {
    if (!locationBundle) return;
    setCoachRefreshing(true);
    setLocErr(null);
    try {
      const { coach } = await coachShooting({
        predicted_class_id: locationBundle.prediction.predicted_class_id,
        predicted_label: locationBundle.prediction.predicted_label,
        class_probabilities: locationBundle.prediction.class_probabilities,
        weather_snapshot: locationBundle.weather_snapshot,
      });
      setLocationBundle({ ...locationBundle, coach });
      if (getNotifyPreference() && notificationPermission() === "granted") {
        notifyResult("Coach updated", "Fresh shooting tips are ready in the app.");
      }
    } catch (e) {
      setLocErr(e instanceof Error ? e.message : "Coach refresh failed");
    } finally {
      setCoachRefreshing(false);
    }
  };

  const toggleNotify = async () => {
    const next = !notifyOn;
    if (!next) {
      setNotifyOn(false);
      setNotifyPreference(false);
      return;
    }
    const location = parseCurrentLocation();
    if (!location) {
      setLocErr("Set valid latitude/longitude and past hours before hourly desktop alerts.");
      return;
    }
    if (!isNotificationApiAvailable()) return;
    const perm = await requestNotificationPermission();
    setNotifPerm(perm);
    if (perm !== "granted") {
      setNotifyOn(false);
      setNotifyPreference(false);
      return;
    }
    setLocErr(null);
    setHourlyNotifyLocation(location);
    setNotifyOn(true);
    setNotifyPreference(true);
  };

  const toggleEmailNotify = async () => {
    const next = !emailNotifyOn;
    const location = parseCurrentLocation();
    if (!location) {
      setLocErr("Set valid latitude/longitude and past hours before email opt-in.");
      return;
    }
    const normalizedEmail = emailAddress.trim().toLowerCase();
    if (!normalizedEmail.includes("@")) {
      setLocErr("Enter a valid email address for hourly updates.");
      return;
    }
    setEmailBusy(true);
    setLocErr(null);
    setEmailSuccess(null);
    try {
      const res = await setEmailSubscription({
        email: normalizedEmail,
        latitude: location.latitude,
        longitude: location.longitude,
        past_hours: location.pastHours,
        enabled: next,
      });
      setEmailNotifyOn(next);
      setEmailNotifyPreference(next);
      setSavedEmail(normalizedEmail);
      if (next) {
        setEmailSuccess(
          res.status === "subscribed"
            ? "Hourly emails are on for this spot — we saved your address and coordinates."
            : "Preferences saved — hourly emails stay aligned with this location.",
        );
      } else {
        setEmailSuccess("Hourly emails are off for this address on the server.");
      }
    } catch (e) {
      setLocErr(e instanceof Error ? e.message : "Email subscription failed");
    } finally {
      setEmailBusy(false);
    }
  };

  useEffect(() => {
    if (!notifyOn) return;
    if (!isNotificationApiAvailable() || notificationPermission() !== "granted") return;

    const msToNextHour = () => {
      const now = Date.now();
      const next = new Date(now);
      next.setMinutes(0, 0, 0);
      next.setMilliseconds(0);
      if (next.getTime() <= now) next.setHours(next.getHours() + 1);
      return Math.max(5_000, next.getTime() - now);
    };

    const tick = async () => {
      const saved = getHourlyNotifyLocation();
      if (!saved) return;
      try {
        const bundle = await predictFromLocation({
          latitude: saved.latitude,
          longitude: saved.longitude,
          past_hours: saved.pastHours,
        });
        const last = getHourlyLastReference();
        if (last === bundle.reference_time_utc) return;
        firePredictionNotification(bundle.prediction, "Hourly location update");
        setHourlyLastReference(bundle.reference_time_utc);
      } catch {
        // keep silent to avoid interrupting active users every hour on transient failures
      }
    };

    let intervalId: number | undefined;
    const timeoutId = window.setTimeout(() => {
      void tick();
      intervalId = window.setInterval(() => void tick(), 60 * 60 * 1000);
    }, msToNextHour());
    return () => {
      window.clearTimeout(timeoutId);
      if (intervalId !== undefined) window.clearInterval(intervalId);
    };
  }, [notifyOn]);

  const caps = status?.capabilities;
  const snap = locationBundle?.weather_snapshot;
  const busy = locLoading || geoLoading;

  return (
    <div className="page">
      <div className="ambient" aria-hidden>
        <MountainBackdrop className="ambient__mountains" />
        <span className="petal petal--1" />
        <span className="petal petal--2" />
        <span className="petal petal--3" />
        <span className="petal petal--4" />
        <span className="petal petal--5" />
      </div>

      <header className="hero">
        <div className="hero__crest">
          <BrandMark size={68} />
          <div className="hero__crest-meta">
            <span className="hero__est">Est. MMXXVI</span>
            <span className="hero__est-rule" aria-hidden />
            <span className="hero__est">Photometric atelier</span>
          </div>
        </div>
        <p className="eyebrow">Études in natural light · 自然光の研究</p>
        <h1 className="display hero__title">
          Lux<span className="hero__title-amp">·</span>Aeterna
        </h1>
        <p className="lede">
          A quiet instrument for photographers — read the sky above any coordinate, receive a verdict
          in plain language, and step out with a tailored set of camera settings for the hour.
        </p>
        <div className="hero__rule" aria-hidden>
          <span className="hero__rule-line" />
          <span className="hero__rule-mark">❀</span>
          <span className="hero__rule-line" />
        </div>
      </header>

      <section className="panel panel--feature">
        <div className="panel__head panel__head--stack">
          <div className="panel__heading">
            <span className="panel__numeral">I</span>
            <div>
              <h2 className="panel__title">Read the light</h2>
              <p className="panel__subtitle">
                Hourly weather and sun geometry, gathered for your coordinates and read by the ensemble — returned as
                a verdict, a snapshot, and a coach.
              </p>
            </div>
          </div>
          <div className="panel__actions">
            <button
              type="button"
              className="btn btn--location"
              onClick={() => void useMyLocationAndAnalyze()}
              disabled={busy || !caps?.predict_event_from_location || !isGeolocationSupported()}
              title={!isGeolocationSupported() ? "Geolocation not available" : "Use device GPS (permission required)"}
            >
              {geoLoading ? "Getting GPS…" : "Use my location"}
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => void runLocationPredict()}
              disabled={busy || !caps?.predict_event_from_location}
            >
              {locLoading ? "Analyzing…" : "Analyze coordinates"}
            </button>
          </div>
        </div>

        {!isGeolocationSupported() && (
          <p className="banner banner--warn small">Geolocation is not supported in this browser. Enter coordinates manually.</p>
        )}
        {!caps?.predict_event_from_location && (
          <p className="banner banner--warn">Load ensemble artifacts on the server to enable location analysis.</p>
        )}

        <div className="notify-row">
          <label className="notify-toggle">
            <input type="checkbox" checked={notifyOn} onChange={() => void toggleNotify()} disabled={notifPerm === "unsupported"} />
            <span>Hourly desktop update for this location (while this tab is open)</span>
          </label>
          <div className="notify-email">
            <input
              className="field__input"
              type="email"
              value={emailAddress}
              onChange={(e) => setEmailAddress(e.target.value)}
              placeholder="you@example.com"
              autoComplete="email"
            />
            <label className="notify-toggle">
              <input type="checkbox" checked={emailNotifyOn} onChange={() => void toggleEmailNotify()} disabled={emailBusy} />
              <span>{emailBusy ? "Saving…" : "Hourly email update for this location"}</span>
            </label>
          </div>
          {notifPerm === "denied" && <span className="small muted">Notifications blocked—enable them in the browser site settings.</span>}
          {notifPerm === "unsupported" && <span className="small muted">Notifications not available in this context.</span>}
          {emailSuccess && (
            <p className="banner banner--ok small" role="status" aria-live="polite">
              {emailSuccess}
            </p>
          )}
          <span className="small muted">Tip: opt-in captures the current lat/lon and past-hours values as your alert location.</span>
        </div>

        <div className="form-grid">
          <label className="field">
            <span className="field__label">Latitude</span>
            <input
              className="field__input mono"
              value={lat}
              onChange={(e) => setLat(e.target.value)}
              inputMode="decimal"
              autoComplete="off"
            />
          </label>
          <label className="field">
            <span className="field__label">Longitude</span>
            <input
              className="field__input mono"
              value={lon}
              onChange={(e) => setLon(e.target.value)}
              inputMode="decimal"
              autoComplete="off"
            />
          </label>
          <label className="field">
            <span className="field__label">Past hours (24–240)</span>
            <input
              className="field__input mono"
              value={pastHours}
              onChange={(e) => setPastHours(e.target.value)}
              inputMode="numeric"
              autoComplete="off"
            />
          </label>
        </div>

        {geoErr && <p className="banner banner--err small">{geoErr}</p>}
        {locErr && <p className="banner banner--err mono small">{locErr}</p>}

        {locationBundle && (
          <div className="results-stack" ref={locationResultsRef}>
            <p className="meta-line mono small">
              <span className="muted">Reference (UTC)</span> {locationBundle.reference_time_utc.slice(0, 19).replace("T", " ")} ·{" "}
              <span className="muted">Coords</span> {locationBundle.latitude.toFixed(4)}, {locationBundle.longitude.toFixed(4)}
            </p>

            {snap && (
              <div className="snap">
                <h3 className="snap__title">Weather at decision time</h3>
                <div className="snap__grid">
                  <div className="snap__cell">
                    <span className="snap__k">Temperature</span>
                    <span className="snap__v mono">{snap.temperature_c?.toFixed(1)} °C</span>
                  </div>
                  <div className="snap__cell">
                    <span className="snap__k">Humidity</span>
                    <span className="snap__v mono">{snap.relative_humidity_pct?.toFixed(0)}%</span>
                  </div>
                  <div className="snap__cell">
                    <span className="snap__k">Visibility</span>
                    <span className="snap__v mono">{(snap.visibility_m / 1000).toFixed(1)} km</span>
                  </div>
                  <div className="snap__cell">
                    <span className="snap__k">Clouds low / mid / high</span>
                    <span className="snap__v mono">
                      {snap.cloud_cover_low_pct?.toFixed(0)} / {snap.cloud_cover_mid_pct?.toFixed(0)} /{" "}
                      {snap.cloud_cover_high_pct?.toFixed(0)}%
                    </span>
                  </div>
                  <div className="snap__cell">
                    <span className="snap__k">Weather code</span>
                    <span className="snap__v mono">{snap.weather_code}</span>
                  </div>
                  <div className="snap__cell">
                    <span className="snap__k">Solar elevation</span>
                    <span className="snap__v mono">{snap.solar_elevation_deg?.toFixed(1)}°</span>
                  </div>
                </div>
              </div>
            )}

            <VerdictCard prediction={locationBundle.prediction} />
            <PredictionBars prediction={locationBundle.prediction} />
            <CoachPanel
              coach={locationBundle.coach}
              onRefreshCoach={caps?.shooting_coach ? () => void refreshCoachOnly() : undefined}
              coachRefreshing={coachRefreshing}
            />
          </div>
        )}
      </section>

      <section className="panel panel--secondary">
        <div className="panel__head panel__head--stack">
          <div className="panel__heading">
            <span className="panel__numeral">II</span>
            <div>
              <h2 className="panel__title">Atelier demo</h2>
              <p className="panel__subtitle">Sends placeholder features straight to the ensemble — no weather call. A quick sanity test.</p>
            </div>
          </div>
          <button type="button" className="btn btn--ghost" onClick={() => void runDemoPredict()} disabled={predLoading || !caps?.lighting_event_ensemble}>
            {predLoading ? "Running…" : "Run demo"}
          </button>
        </div>
        {!caps?.lighting_event_ensemble && <p className="banner banner--warn small">Ensemble not loaded.</p>}
        {predErr && <p className="banner banner--err mono small">{predErr}</p>}
        {prediction && (
          <div className="results-stack">
            <VerdictCard prediction={prediction} />
            <PredictionBars prediction={prediction} />
          </div>
        )}
      </section>

      <details className="panel panel--details">
        <summary className="details-summary">
          <span className="panel__numeral panel__numeral--inline">III</span>
          The instrument · status &amp; capabilities
        </summary>
        <div className="details-body">
          <div className="panel__head">
            <span className="muted small">Backend health and loaded routes</span>
            <button type="button" className="btn btn--ghost btn--sm" onClick={() => void refresh()} disabled={loading}>
              {loading ? "…" : "Refresh"}
            </button>
          </div>
          {err && <p className="banner banner--err">{err}</p>}
          {!err && status && (
            <div className="grid">
              <div className="stat">
                <span className="stat__label">Health</span>
                <span className={`stat__value ${health === "healthy" ? "ok" : ""}`}>{health ?? "—"}</span>
              </div>
              <div className="stat">
                <span className="stat__label">LSTM (legacy)</span>
                <span className="stat__value mono">{status.lstm_model_version}</span>
              </div>
              <div className="stat">
                <span className="stat__label">MLP (legacy)</span>
                <span className="stat__value mono">{status.mlp_model_version}</span>
              </div>
              <div className="stat">
                <span className="stat__label">Parquet freshness</span>
                <span className="stat__value">
                  {status.data_freshness_minutes == null ? "—" : `${status.data_freshness_minutes.toFixed(0)} min`}
                </span>
              </div>
            </div>
          )}
          {caps && (
            <div className="caps">
              <CapabilityPill label="ALQS /predict" on={caps.legacy_alqs_predict} />
              <CapabilityPill label="/forecast" on={caps.legacy_forecast} />
              <CapabilityPill label="/recommend" on={caps.legacy_recommend} />
              <CapabilityPill label="/predict/event" on={caps.lighting_event_ensemble} />
              <CapabilityPill label="/from_location" on={caps.predict_event_from_location ?? false} />
              <CapabilityPill label="/coach/shooting" on={caps.shooting_coach ?? false} />
            </div>
          )}
        </div>
      </details>

      <footer className="foot">
        <div className="foot__seal" aria-hidden>
          <BrandMark size={32} variant="minimal" />
        </div>
        <p className="foot__line">
          <span className="display">Lux Aeterna</span>
          <span className="muted"> — eternal light.</span>
        </p>
        <p className="foot__meta small muted">
          Weather by Open-Meteo · solar geometry by PyEphem ·{" "}
          <a href="/api/docs" target="_blank" rel="noreferrer">
            API documentation
          </a>
        </p>
      </footer>
    </div>
  );
}
