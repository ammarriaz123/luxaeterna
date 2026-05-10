"""Rules-first shooting coach; optional OpenAI narrative when COACH_LLM=1 and OPENAI_API_KEY is set."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from api.schemas import CoachRecommendation, LightingClassScore

LOGGER = logging.getLogger("luxaeterna.api.coach")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"


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


def _coach_llm_enabled() -> bool:
    return bool(os.getenv("OPENAI_API_KEY")) and os.getenv("COACH_LLM", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def maybe_enrich_coach_with_openai(
    base: CoachRecommendation,
    *,
    class_probabilities: list[LightingClassScore],
    weather_snapshot: dict[str, Any],
) -> CoachRecommendation:
    if not _coach_llm_enabled():
        return base

    key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("COACH_OPENAI_MODEL", "gpt-4o-mini")
    try:
        probs_line = ", ".join(f"{c.label}:{c.probability:.2f}" for c in class_probabilities)
        user = (
            f"Predicted lighting event: {base.predicted_label}. "
            f"Class probabilities: {probs_line}. "
            f"Latest weather snapshot JSON: {json.dumps(weather_snapshot)}. "
            "You are a photography coach. In 3–5 short bullet sentences, add creative shooting advice "
            "(composition, exposure mindset, one lens idea). No equipment sales. Plain text only, no markdown."
        )
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You assist photographers with concise, practical advice. Output plain text only.",
                },
                {"role": "user", "content": user},
            ],
            "max_tokens": 300,
            "temperature": 0.6,
        }
        with httpx.Client(timeout=60.0) as client:
            r = client.post(
                OPENAI_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        text = str(text).strip()
        if not text:
            return base
        return base.model_copy(
            update={
                "source": "rules+openai",
                "llm_addon": text,
            }
        )
    except Exception as exc:
        LOGGER.warning("OpenAI coach enrichment failed: %s", exc)
        return base
