"""Rules-first shooting coach with optional LLM enrichment.

Uses any OpenAI-compatible Chat Completions API (OpenAI, Groq, etc.) via env.
Structured JSON merge (scenario-conditioned) + optional SSE stream for narrative.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterator
from typing import Any

import httpx
from pydantic import BaseModel, Field

from api.schemas import CoachAnchoredContext, CoachChatRequest, CoachRecommendation, LightingClassScore

LOGGER = logging.getLogger("luxaeterna.api.coach")

_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

_MAX_FIELD_LEN = 1200
_MAX_ADDON = 2800
_MAX_CHECK_ITEMS = 4
_MAX_CHECK_ITEM_LEN = 160

try:
    from typing import TypedDict

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langgraph.graph import END, StateGraph

    _LANGGRAPH_AVAILABLE = True
except Exception:
    _LANGGRAPH_AVAILABLE = False
    TypedDict = dict  # type: ignore[assignment,misc]


class CoachRefinementDraft(BaseModel):
    """LLM output shape for structured merge (sync path includes llm_addon)."""

    shooting_mode: str | None = None
    iso_suggestion: str | None = None
    aperture_guidance: str | None = None
    shutter_guidance: str | None = None
    white_balance: str | None = None
    gear_notes: str | None = None
    creative_brief: str | None = None
    checklist_append: list[str] = Field(default_factory=list)
    llm_addon: str | None = None


def build_rules_coach(
    *,
    predicted_label: str,
    predicted_class_id: int,
    class_probabilities: list[LightingClassScore],
    weather_snapshot: dict[str, Any],
) -> CoachRecommendation:
    """Heuristic packs per ensemble class — safe defaults, not a substitute for judgment."""
    se = float(weather_snapshot.get("solar_elevation_deg", 0.0))
    low = float(weather_snapshot.get("cloud_cover_low_pct", 0.0))
    mid = float(weather_snapshot.get("cloud_cover_mid_pct", 0.0))

    base_checklist = [
        "Charge batteries; spare card.",
        "Clean front element / sensor check.",
        "Scout foreground while light is still flat.",
    ]

    if predicted_class_id == 0:
        return CoachRecommendation(
            predicted_label=predicted_label,
            shooting_mode="Flexible (Aperture or Program)",
            iso_suggestion="ISO 100–800 depending on wind/handholding",
            aperture_guidance="ƒ/5.6–ƒ/11 for general scenes; wider for subject isolation.",
            shutter_guidance="1/125s+ for handheld; tripod if blending exposures later.",
            white_balance="Auto or Daylight; tweak for mood in post.",
            gear_notes="Polarizer optional if sky has glare at non-wide angles.",
            checklist=base_checklist
            + [
                "Flat light: look for texture, color blocks, minimal compositions.",
            ],
            creative_brief="No strong golden/diffusion signal — prioritize storytelling and local contrast in editing.",
            source="rules",
            llm_addon=None,
        )

    if predicted_class_id == 1:
        return CoachRecommendation(
            predicted_label=predicted_label,
            shooting_mode="Aperture priority or Manual",
            iso_suggestion="ISO 100–400 (keep noise down; tripod if needed)",
            aperture_guidance="ƒ/2.8–ƒ/5.6 for subject separation; stop down if you need more depth.",
            shutter_guidance="Watch highlights on skin/sky — bias toward underexposure 1/3 stop, lift shadows later.",
            white_balance="Daylight or Cloudy for warmth; bracket WB if unsure.",
            gear_notes="Lens hood; consider 85mm / 50mm for portraits; wide for environment context.",
            checklist=base_checklist
            + [
                f"Low sun (~{se:.1f}°): long shadows — use them as leading lines.",
                "Shoot both toward and away from the sun for different palette.",
            ],
            creative_brief="Golden-hour lean: warm low-angle light. Seek rim light and side-lit texture.",
            source="rules",
            llm_addon=None,
        )

    if predicted_class_id == 2:
        return CoachRecommendation(
            predicted_label=predicted_label,
            shooting_mode="Manual or Aperture priority",
            iso_suggestion="ISO 100–800; bump only if wind shakes foliage.",
            aperture_guidance="ƒ/8–ƒ/11 for landscape depth; polarizer can deepen sky if angle works.",
            shutter_guidance="Faster if clouds are moving and you want crisp structure; slower for soft blur.",
            white_balance="Cloudy or Auto; diffusion can swing magenta/green — shoot RAW.",
            gear_notes=f"Layered clouds (low {low:.0f}%, mid {mid:.0f}%): CPL + tripod for ND blends if needed.",
            checklist=base_checklist
            + [
                "Expose for highlights; recover shadow in RAW.",
                "Look for god-rays / cloud texture at edges of fronts.",
            ],
            creative_brief="Diffusion-forward: soft wrap light, drama in sky. Think mood, texture, minimal color palette.",
            source="rules",
            llm_addon=None,
        )

    # class 3 — golden + diffusion
    return CoachRecommendation(
        predicted_label=predicted_label,
        shooting_mode="Manual (tripod strongly recommended)",
        iso_suggestion="ISO 100–400; keep ISO low for blending / HDR brackets.",
        aperture_guidance="ƒ/8–ƒ/11 for landscapes; wider if foreground subject is close.",
        shutter_guidance="Bracket ±1–2 EV if dynamic range spikes; watch clipping on sunlit cloud edges.",
        white_balance="Cloudy or Daylight test shots; mixed sources — RAW mandatory.",
        gear_notes="Tripod, remote, CPL, soft grad ND if horizon is bright. Lens hood against flare.",
        checklist=base_checklist
        + [
            "Compose with sun position + cloud texture — avoid chaotic merges at the horizon.",
            "Shoot a burst as the gap evolves; light changes fast.",
        ],
        creative_brief="Rare mix: warm low sun plus textured sky. Prioritize safety (don’t stare into sun), then bracket.",
        source="rules",
        llm_addon=None,
    )


def _coach_llm_api_key() -> str:
    """Prefer explicit coach key, then OpenAI-compatible env names, then Groq."""
    return (
        os.getenv("COACH_LLM_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("GROQ_API_KEY", "").strip()
    )


def _coach_llm_base_url() -> str:
    """OpenAI-compatible API root (no trailing slash), e.g. https://api.openai.com/v1 or Groq."""
    explicit = os.getenv("COACH_LLM_BASE_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    key = _coach_llm_api_key()
    if os.getenv("GROQ_API_KEY", "").strip() or key.startswith("gsk_"):
        return "https://api.groq.com/openai/v1"
    return "https://api.openai.com/v1"


def _is_groq_base(url: str) -> bool:
    return "groq.com" in url.lower()


def _coach_llm_model(base_url: str) -> str:
    raw = (os.getenv("COACH_OPENAI_MODEL") or os.getenv("COACH_LLM_MODEL") or "").strip()
    if not raw:
        return _DEFAULT_GROQ_MODEL if _is_groq_base(base_url) else _DEFAULT_OPENAI_MODEL
    if _is_groq_base(base_url) and raw in ("gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"):
        return _DEFAULT_GROQ_MODEL
    return raw


def _chat_completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def _coach_llm_enabled() -> bool:
    return bool(_coach_llm_api_key()) and os.getenv("COACH_LLM", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _coach_backend() -> str:
    raw = os.getenv("COACH_LLM_BACKEND", "openai_http").strip().lower()
    if raw in {"langgraph", "langchain"}:
        return "langgraph"
    return "openai_http"


def _winner_probability(class_probabilities: list[LightingClassScore], predicted_class_id: int) -> float:
    for c in class_probabilities:
        if c.class_id == predicted_class_id:
            return float(c.probability)
    return max((float(c.probability) for c in class_probabilities), default=0.0)


def _confidence_lane(p: float) -> str:
    if p >= 0.55:
        return "high"
    if p >= 0.38:
        return "medium"
    return "low"


def _scenario_system_prompt(*, predicted_label: str, predicted_class_id: int, lane: str) -> str:
    """Scenario + confidence lane — same API key, richer conditioning."""
    label_key = predicted_label.strip().lower().replace(" ", "_")
    base = (
        "You refine a rules-first photography coach for LuxAeterna. "
        "Obey JSON only. Never invent numeric weather values; only interpret the JSON snapshot provided. "
        "No equipment sales, no markdown, no URLs. Keep each string concise and practical."
    )
    scene = {
        0: "Scene: flat or ambiguous light — emphasize texture, restraint, and editorial seeing.",
        1: "Scene: golden-hour lean — emphasize warm low sun, rim light, and directional composition.",
        2: "Scene: diffusion / drama sky — emphasize soft wrap, cloud texture, exposure discipline.",
        3: "Scene: golden hour plus diffusion — emphasize rare light, safety near sun, bracketing discipline.",
    }.get(predicted_class_id, "Scene: general outdoor light.")
    lane_note = {
        "high": "The model is fairly decisive — align tone with confidence, avoid hedging every sentence.",
        "medium": "Probabilities are mixed — acknowledge uncertainty once, still give crisp choices.",
        "low": "Probabilities are close — stress scouting, small tests, and reversible decisions.",
    }.get(lane, lane)
    return f"{base} {scene} Predicted label token: {predicted_label} ({label_key}). {lane_note}"


def _json_output_instructions(*, include_llm_addon: bool) -> str:
    keys = (
        '"shooting_mode","iso_suggestion","aperture_guidance","shutter_guidance","white_balance",'
        '"gear_notes","creative_brief","checklist_append"'
    )
    addon = ', "llm_addon"' if include_llm_addon else ""
    checklist = (
        '"checklist_append": string array, 0–3 short new checklist lines not already implied '
        f'(each ≤{_MAX_CHECK_ITEM_LEN} chars)'
    )
    out = (
        f"Return a single JSON object with keys {keys}{addon}. "
        "Use null for any key you are not improving. "
        f"{checklist}. "
    )
    if include_llm_addon:
        out += (
            f'"llm_addon": 3–6 sentences of polished, plain-text creative guidance that complements '
            f'(not repeats verbatim) the rules card; ≤{_MAX_ADDON} characters; no markdown.'
        )
    else:
        out += "Do not include llm_addon."
    return out


def _build_structured_user_message(
    *,
    base: CoachRecommendation,
    class_probabilities: list[LightingClassScore],
    weather_snapshot: dict[str, Any],
    include_llm_addon: bool,
) -> str:
    probs_line = ", ".join(f"{c.label}:{c.probability:.2f}" for c in class_probabilities)
    coach_card = base.model_dump(mode="json")
    return (
        f"{_json_output_instructions(include_llm_addon=include_llm_addon)}\n\n"
        f"Rules coach JSON: {json.dumps(coach_card)}\n"
        f"Class probabilities: {probs_line}\n"
        f"Weather snapshot JSON: {json.dumps(weather_snapshot)}\n"
    )


def _strip_json_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _parse_refinement_json(raw: str) -> CoachRefinementDraft | None:
    try:
        data = json.loads(_strip_json_fences(raw))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    try:
        return CoachRefinementDraft.model_validate(data)
    except Exception:
        return None


def _clamp_str(value: str | None, max_len: int) -> str | None:
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    return v[:max_len]


def _merge_refinement(base: CoachRecommendation, draft: CoachRefinementDraft) -> CoachRecommendation:
    updates: dict[str, Any] = {}
    changed = False
    for field in (
        "shooting_mode",
        "iso_suggestion",
        "aperture_guidance",
        "shutter_guidance",
        "white_balance",
        "gear_notes",
        "creative_brief",
    ):
        raw = getattr(draft, field)
        clamped = _clamp_str(raw, _MAX_FIELD_LEN)
        if clamped and clamped != getattr(base, field):
            updates[field] = clamped
            changed = True

    addon = _clamp_str(draft.llm_addon, _MAX_ADDON)
    if addon:
        updates["llm_addon"] = addon
        changed = True

    merged_check = list(base.checklist)
    seen = {s.strip().lower() for s in merged_check}
    for item in draft.checklist_append[:_MAX_CHECK_ITEMS]:
        line = _clamp_str(item, _MAX_CHECK_ITEM_LEN)
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        merged_check.append(line)
    if len(merged_check) != len(base.checklist):
        updates["checklist"] = merged_check
        changed = True

    if not changed:
        return base
    updates["source"] = "rules+openai"
    return base.model_copy(update=updates)


def _post_chat_completion_json(
    *,
    system_prompt: str,
    user_message: str,
    include_response_format: bool,
) -> str:
    api_root = _coach_llm_base_url()
    key = _coach_llm_api_key()
    model = _coach_llm_model(api_root)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 900 if include_response_format else 500,
        "temperature": 0.45,
    }
    if include_response_format:
        payload["response_format"] = {"type": "json_object"}

    with httpx.Client(timeout=90.0) as client:
        r = client.post(
            _chat_completions_url(api_root),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
        )
        r.raise_for_status()
        data = r.json()
    return str((data.get("choices") or [{}])[0].get("message", {}).get("content") or "").strip()


def _enrich_with_structured_http_fixed_id(
    *,
    base: CoachRecommendation,
    predicted_class_id: int,
    class_probabilities: list[LightingClassScore],
    weather_snapshot: dict[str, Any],
    include_llm_addon: bool,
) -> CoachRecommendation:
    p = _winner_probability(class_probabilities, predicted_class_id)
    lane = _confidence_lane(p)
    system = _scenario_system_prompt(
        predicted_label=base.predicted_label,
        predicted_class_id=predicted_class_id,
        lane=lane,
    )
    user_msg = _build_structured_user_message(
        base=base,
        class_probabilities=class_probabilities,
        weather_snapshot=weather_snapshot,
        include_llm_addon=include_llm_addon,
    )
    raw = _post_chat_completion_json(
        system_prompt=system,
        user_message=user_msg,
        include_response_format=True,
    )
    draft = _parse_refinement_json(raw)
    if draft is None:
        LOGGER.warning("Coach structured parse failed; raw snippet: %s", raw[:200])
        return base
    if not include_llm_addon:
        draft = draft.model_copy(update={"llm_addon": None})
    return _merge_refinement(base, draft)


def _enrich_with_langgraph(
    *,
    base: CoachRecommendation,
    predicted_class_id: int,
    class_probabilities: list[LightingClassScore],
    weather_snapshot: dict[str, Any],
    include_llm_addon: bool,
) -> CoachRecommendation:
    if not _LANGGRAPH_AVAILABLE:
        raise RuntimeError("LangGraph/LangChain packages not installed")

    api_root = _coach_llm_base_url()
    model = _coach_llm_model(api_root)
    key = _coach_llm_api_key()
    pred_id = predicted_class_id
    p = _winner_probability(class_probabilities, pred_id)
    lane = _confidence_lane(p)
    system = _scenario_system_prompt(
        predicted_label=base.predicted_label,
        predicted_class_id=pred_id,
        lane=lane,
    )
    user_msg = _build_structured_user_message(
        base=base,
        class_probabilities=class_probabilities,
        weather_snapshot=weather_snapshot,
        include_llm_addon=include_llm_addon,
    )

    class CoachGraphState(TypedDict):
        system: str
        user_message: str
        llm_output: str

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system}"),
            ("user", "{user_message}"),
        ]
    )
    llm = ChatOpenAI(
        model=model,
        api_key=key,
        base_url=api_root,
        temperature=0.45,
        max_tokens=900,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    def draft_node(state: CoachGraphState) -> CoachGraphState:
        chain = prompt | llm
        response = chain.invoke({"system": state["system"], "user_message": state["user_message"]})
        return {
            "system": state["system"],
            "user_message": state["user_message"],
            "llm_output": str(getattr(response, "content", "")).strip(),
        }

    graph = StateGraph(CoachGraphState)
    graph.add_node("draft", draft_node)
    graph.set_entry_point("draft")
    graph.add_edge("draft", END)
    app = graph.compile()

    out = app.invoke({"system": system, "user_message": user_msg, "llm_output": ""})
    raw = str(out.get("llm_output", "")).strip()
    draft = _parse_refinement_json(raw)
    if draft is None:
        LOGGER.warning("Coach LangGraph parse failed; raw snippet: %s", raw[:200])
        return base
    if not include_llm_addon:
        draft = draft.model_copy(update={"llm_addon": None})
    return _merge_refinement(base, draft)


def _run_structured_enrich(
    base: CoachRecommendation,
    *,
    predicted_class_id: int,
    class_probabilities: list[LightingClassScore],
    weather_snapshot: dict[str, Any],
    include_llm_addon: bool,
) -> CoachRecommendation:
    backend = _coach_backend()
    if backend == "langgraph":
        return _enrich_with_langgraph(
            base=base,
            predicted_class_id=predicted_class_id,
            class_probabilities=class_probabilities,
            weather_snapshot=weather_snapshot,
            include_llm_addon=include_llm_addon,
        )
    return _enrich_with_structured_http_fixed_id(
        base=base,
        predicted_class_id=predicted_class_id,
        class_probabilities=class_probabilities,
        weather_snapshot=weather_snapshot,
        include_llm_addon=include_llm_addon,
    )


def maybe_enrich_coach_with_openai(
    base: CoachRecommendation,
    *,
    predicted_class_id: int,
    class_probabilities: list[LightingClassScore],
    weather_snapshot: dict[str, Any],
) -> CoachRecommendation:
    if not _coach_llm_enabled():
        return base

    try:
        merged = _run_structured_enrich(
            base,
            predicted_class_id=predicted_class_id,
            class_probabilities=class_probabilities,
            weather_snapshot=weather_snapshot,
            include_llm_addon=True,
        )
        if merged.source == "rules":
            return base
        return merged
    except Exception as exc:
        LOGGER.warning("Coach enrichment failed with backend '%s': %s", _coach_backend(), exc)
        return base


def _stream_narrative_tokens(
    *,
    base: CoachRecommendation,
    class_probabilities: list[LightingClassScore],
    weather_snapshot: dict[str, Any],
    predicted_class_id: int,
) -> Iterator[str]:
    """Stream plain-text narrative tokens (OpenAI-compatible SSE)."""
    api_root = _coach_llm_base_url()
    key = _coach_llm_api_key()
    model = _coach_llm_model(api_root)
    p = _winner_probability(class_probabilities, predicted_class_id)
    lane = _confidence_lane(p)
    system = (
        _scenario_system_prompt(
            predicted_label=base.predicted_label,
            predicted_class_id=predicted_class_id,
            lane=lane,
        )
        + " Write only flowing prose (no JSON, no markdown). "
        "4–6 sentences: mood, composition, exposure mindset, one lens or focal length idea. "
        "Ground every claim in the coach card and snapshot; do not invent numbers."
    )
    user_msg = (
        "Coach card after structured refinement:\n"
        f"{json.dumps(base.model_dump(mode='json'))}\n\n"
        f"Weather snapshot JSON:\n{json.dumps(weather_snapshot)}\n"
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 450,
        "temperature": 0.55,
        "stream": True,
    }

    with httpx.Client(timeout=120.0) as client:
        with client.stream(
            "POST",
            _chat_completions_url(api_root),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
        ) as r:
            r.raise_for_status()
            for raw_line in r.iter_lines():
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line or not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {}).get("content")
                if delta:
                    yield str(delta)


def iter_coach_shooting_sse(
    *,
    predicted_class_id: int,
    predicted_label: str,
    class_probabilities: list[LightingClassScore],
    weather_snapshot: dict[str, Any],
) -> Iterator[bytes]:
    """SSE stream: rules coach → structured merge (no llm_addon) → narrative tokens → final coach JSON."""

    def pack(obj: dict[str, Any]) -> bytes:
        return f"data: {json.dumps(obj, default=str)}\n\n".encode("utf-8")

    base = build_rules_coach(
        predicted_label=predicted_label,
        predicted_class_id=predicted_class_id,
        class_probabilities=class_probabilities,
        weather_snapshot=weather_snapshot,
    )
    yield pack({"type": "rules", "coach": base.model_dump(mode="json")})

    if not _coach_llm_enabled():
        yield pack({"type": "final", "coach": base.model_dump(mode="json")})
        return

    try:
        merged = _run_structured_enrich(
            base,
            predicted_class_id=predicted_class_id,
            class_probabilities=class_probabilities,
            weather_snapshot=weather_snapshot,
            include_llm_addon=False,
        )
        if merged.llm_addon:
            merged = merged.model_copy(update={"llm_addon": None})
        yield pack({"type": "structured", "coach": merged.model_dump(mode="json")})

        pieces: list[str] = []
        for token in _stream_narrative_tokens(
            base=merged,
            class_probabilities=class_probabilities,
            weather_snapshot=weather_snapshot,
            predicted_class_id=predicted_class_id,
        ):
            pieces.append(token)
            yield pack({"type": "token", "text": token})

        addon = "".join(pieces).strip()
        if len(addon) > _MAX_ADDON:
            addon = addon[:_MAX_ADDON]
        final = merged.model_copy(
            update={
                "llm_addon": addon or None,
                "source": "rules+openai" if addon else merged.source,
            }
        )
        yield pack({"type": "final", "coach": final.model_dump(mode="json")})
    except Exception as exc:
        LOGGER.warning("Coach stream failed: %s", exc)
        yield pack({"type": "error", "message": str(exc)})
        yield pack({"type": "final", "coach": base.model_dump(mode="json")})


def coach_llm_available() -> bool:
    """True when COACH_LLM is on and an API key is present (for gating chat endpoints)."""
    return _coach_llm_enabled()


def _anchor_bundle_json(ctx: CoachAnchoredContext) -> str:
    """Serialize anchored context for the system prompt."""
    payload = {
        "latitude": ctx.latitude,
        "longitude": ctx.longitude,
        "reference_time_utc": ctx.reference_time_utc,
        "predicted_label": ctx.predicted_label,
        "predicted_class_id": ctx.predicted_class_id,
        "class_probabilities": [c.model_dump(mode="json") for c in ctx.class_probabilities],
        "weather_snapshot": ctx.weather_snapshot,
        "coach_card": ctx.coach.model_dump(mode="json"),
    }
    return json.dumps(payload, default=str)


def _anchor_chat_system(ctx: CoachAnchoredContext) -> str:
    return (
        "You are the Lux Æterna atelier coach in the Salle de conseil. You speak only for the "
        "single anchored forecast encoded in ANCHOR_JSON below — this coordinate, this reference hour, "
        "these probabilities and weather numbers, this coach card.\n"
        "Rules: Ground every factual claim in ANCHOR_JSON. If the visitor describes another place, "
        "another time, or asks for a new forecast, politely refuse and invite them to run a fresh "
        "analysis there; never fabricate numbers.\n"
        "Tone: warm, precise, unhurried — like a maître de chapelle of light, not a sales blog.\n"
        "Format: plain prose; short paragraphs; you may use light dashes or one short bullet list "
        "when it clarifies action. No markdown code fences; no JSON in replies.\n\n"
        f"ANCHOR_JSON:\n{_anchor_bundle_json(ctx)}"
    )


def iter_anchor_chat_sse(request: CoachChatRequest) -> Iterator[bytes]:
    """SSE: optional start pulse, streamed assistant tokens, done or error."""

    def pack(obj: dict[str, Any]) -> bytes:
        return f"data: {json.dumps(obj, default=str)}\n\n".encode("utf-8")

    if not _coach_llm_enabled():
        yield pack(
            {
                "type": "error",
                "message": "Coach LLM is not enabled on the server (COACH_LLM and an API key).",
            }
        )
        return

    api_root = _coach_llm_base_url()
    key = _coach_llm_api_key()
    model = _coach_llm_model(api_root)

    system = _anchor_chat_system(request.context)
    openai_messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for turn in request.messages:
        openai_messages.append({"role": turn.role, "content": turn.content})

    payload: dict[str, Any] = {
        "model": model,
        "messages": openai_messages,
        "max_tokens": 900,
        "temperature": 0.55,
        "stream": True,
    }

    yield pack({"type": "start"})

    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream(
                "POST",
                _chat_completions_url(api_root),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
            ) as r:
                r.raise_for_status()
                for raw_line in r.iter_lines():
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if not line or not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ").strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {}).get("content")
                    if delta:
                        yield pack({"type": "token", "text": str(delta)})
        yield pack({"type": "done"})
    except Exception as exc:
        LOGGER.warning("Anchored chat stream failed: %s", exc)
        yield pack({"type": "error", "message": str(exc)})
