import { useCallback, useEffect, useRef, useState } from "react";
import {
  coachAnchoredChatStream,
  type PredictFromLocationResponse,
  type SalleChatStreamEvent,
} from "./api";

type ChatTurn = { role: "user" | "assistant"; content: string };

const STARTERS = [
  "How would you compose toward the brightest part of the scene?",
  "What focal length would you reach for first in this light?",
  "If the sky shifts in the next thirty minutes, what should I watch first?",
] as const;

function buildContext(bundle: PredictFromLocationResponse) {
  return {
    latitude: bundle.latitude,
    longitude: bundle.longitude,
    reference_time_utc: bundle.reference_time_utc,
    predicted_label: bundle.prediction.predicted_label,
    predicted_class_id: bundle.prediction.predicted_class_id,
    class_probabilities: bundle.prediction.class_probabilities,
    weather_snapshot: bundle.weather_snapshot as Record<string, number>,
    coach: bundle.coach,
  };
}

export function CoachSalle({
  bundle,
  coachLlmEnabled,
}: {
  bundle: PredictFromLocationResponse;
  coachLlmEnabled: boolean;
}) {
  const [messages, setMessages] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [streamText, setStreamText] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const streamAcc = useRef("");
  const streamErrRef = useRef<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  let refIso = bundle.reference_time_utc.trim();
  if (!/[zZ]$|[+-]\d{2}/.test(refIso)) {
    refIso = refIso.includes("T") ? `${refIso}Z` : `${refIso.replace(" ", "T")}Z`;
  }
  const refParsed = new Date(refIso);
  const refUtcDisplay = Number.isNaN(refParsed.getTime())
    ? bundle.reference_time_utc
    : `${refParsed.toISOString().slice(0, 19).replace("T", " ")} UTC`;
  const refLocalDisplay = Number.isNaN(refParsed.getTime())
    ? ""
    : refParsed.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

  const scrollToEnd = useCallback(() => {
    requestAnimationFrame(() => endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }));
  }, []);

  useEffect(() => {
    scrollToEnd();
  }, [messages, streamText, scrollToEnd]);

  const runAssistant = useCallback(
    async (history: ChatTurn[]) => {
      streamAcc.current = "";
      setStreamText("");
      setErr(null);
      streamErrRef.current = null;
      await coachAnchoredChatStream(
        { messages: history, context: buildContext(bundle) },
        (ev: SalleChatStreamEvent) => {
          if (ev.type === "token") {
            streamAcc.current += ev.text;
            setStreamText(streamAcc.current);
          }
          if (ev.type === "error") {
            streamErrRef.current = ev.message;
            streamAcc.current = "";
            setStreamText("");
          }
        },
      );
      const reply = streamAcc.current.trim();
      streamAcc.current = "";
      setStreamText("");
      const se = streamErrRef.current;
      streamErrRef.current = null;
      if (se) {
        setErr(se);
        return;
      }
      if (reply) {
        setMessages((m) => [...m, { role: "assistant", content: reply }]);
      }
    },
    [bundle],
  );

  const send = async (raw: string) => {
    const text = raw.trim();
    if (!text || busy || !coachLlmEnabled) return;
    setBusy(true);
    setErr(null);
    const userTurn: ChatTurn = { role: "user", content: text };
    const nextHistory = [...messages, userTurn];
    setMessages(nextHistory);
    setInput("");
    try {
      await runAssistant(nextHistory);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "The atelier could not answer.");
      setMessages((m) => m.slice(0, -1));
    } finally {
      setBusy(false);
    }
  };

  const clearThread = () => {
    if (busy) return;
    setMessages([]);
    setInput("");
    setErr(null);
    setStreamText("");
    streamAcc.current = "";
  };

  return (
    <aside className="salle" aria-label="Anchored coach dialogue for this forecast hour">
      <div className="salle__frame">
        <div className="salle__head">
          <span className="salle__mark" aria-hidden>
            ❀
          </span>
          <div>
            <h3 className="salle__title">Salle de conseil</h3>
            <p className="salle__subtitle">
              The coach stays inside the <strong>forecast hour this run used</strong>. That is the hour on the model’s
              timeline, not necessarily the minute you pressed analyze.
              <span className="mono"> {refUtcDisplay}</span>
              {refLocalDisplay ? (
                <>
                  {" "}
                  <span className="salle__dot" aria-hidden>
                    ·
                  </span>{" "}
                  here <span className="mono">{refLocalDisplay}</span>
                </>
              ) : null}
              . The verdict and weather card above stay the source of truth.
            </p>
          </div>
        </div>

        {!coachLlmEnabled && (
          <p className="banner banner--warn small">
            The atelier door is latched: enable <span className="mono">COACH_LLM</span> and an API key on the server to
            speak with the anchored coach.
          </p>
        )}

        {coachLlmEnabled && (
          <>
            <div className="salle__starters" aria-label="Suggested openings">
              {STARTERS.map((q) => (
                <button
                  key={q}
                  type="button"
                  className="salle__chip"
                  disabled={busy}
                  onClick={() => void send(q)}
                >
                  {q}
                </button>
              ))}
            </div>

            <p className="salle__field-hint muted small">
              Replies gather in the transcript. <strong>Your message</strong> is the box below.
            </p>

            <div className="salle__transcript-wrap">
              <p className="salle__transcript-label">Transcript</p>
              <div
                className={`salle__thread${messages.length === 0 && !busy && !streamText ? " salle__thread--empty" : ""}`}
                role="log"
                aria-live="polite"
                aria-label="Conversation transcript"
              >
                {messages.length === 0 && !busy && !streamText && (
                  <p className="salle__thread-hint muted small">No messages yet. Pick a starter or write below.</p>
                )}
                {messages.map((m, i) => (
                  <div key={i} className={`salle__msg salle__msg--${m.role}`}>
                    <span className="salle__msg-label">{m.role === "user" ? "You" : "Atelier"}</span>
                    <p className="salle__msg-body">{m.content}</p>
                  </div>
                ))}
                {busy && streamText && (
                  <div className="salle__msg salle__msg--assistant salle__msg--stream">
                    <span className="salle__msg-label">Atelier</span>
                    <p className="salle__msg-body">{streamText}</p>
                  </div>
                )}
                <div ref={endRef} />
              </div>
            </div>

            {err && <p className="banner banner--err small">{err}</p>}

            <form
              className="salle__form"
              onSubmit={(e) => {
                e.preventDefault();
                void send(input);
              }}
            >
              <label className="salle__compose-label" htmlFor="salle-input">
                Your message
              </label>
              <textarea
                id="salle-input"
                className="salle__input"
                rows={3}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type here, then Send — questions must stay inside this spot and forecast hour."
                disabled={busy}
                maxLength={3500}
              />
              <div className="salle__form-actions">
                <button type="button" className="btn btn--ghost btn--sm" onClick={clearThread} disabled={busy || messages.length === 0}>
                  Clear thread
                </button>
                <button type="submit" className="btn" disabled={busy || !input.trim()}>
                  {busy ? "Listening…" : "Send"}
                </button>
              </div>
            </form>
          </>
        )}
      </div>
    </aside>
  );
}
