"""
Part 7 Quality Evaluation & Failure Mode Analysis Module for PromoAI.
Explicitly identifies where the AI tool picks badly:
- WEAK_HOOK: First 3 seconds are a slow atmospheric camera pan or lack arresting dialogue/action
- WEAK_CLIFFHANGER: Ends on a resolved beat or low-tension statement
- MISSING_STANDALONE_CONTEXT: Relies on mid-season lore, secondary names, or procedural logistics
- INVALID_DURATION: Combined duration outside [20s, 60s]
- TIMESTAMP_DRIFT_OR_UNVERIFIED: Snapped or unverified timestamps
- UNSUITABLE_FOR_PROMO: Low overall dramatic tension or transitional filler scene
Also manages Human-in-the-Loop quality ratings ('Good', 'Average', 'Bad').
"""
import json
import os
from typing import List, Dict, Any
from src.config import AppConfig
from src.models import EpisodeDocument, PromoCandidate


def audit_candidate_quality(
    cand: PromoCandidate,
    episodes_by_num: Dict[int, EpisodeDocument],
    config: AppConfig,
) -> List[str]:
    """
    Runs deterministic and rubric-based quality checks on a PromoCandidate to detect
    where the tool picks badly or requires human editor attention.
    """
    flags: List[str] = list(cand.quality_flags)

    # 1. Check first 3 seconds of the opening segment for a slow camera pan / weak hook
    if cand.segments:
        first_seg = cand.segments[0]
        ep_doc = episodes_by_num.get(first_seg.episode_number)
        if ep_doc:
            first_3s_events = [
                ev for ev in ep_doc.events
                if ev.end_sec > first_seg.start_sec and ev.start_sec <= first_seg.start_sec + 3.2
            ]
            has_dialogue_or_action_in_3s = any(
                ev.event_type in ("DIALOGUE", "ACTION") for ev in first_3s_events
            )
            if not has_dialogue_or_action_in_3s:
                flags.append("WEAK_OPENING_3S_SLOW_CAMERA")
                cand.review_notes.append(
                    "Quality Audit: First 3.0s contain only a CAMERA shot without immediate dialogue or physical action; risk of viewer scroll-past."
                )

    # 2. Check sub-score thresholds for explicit failure modes
    if cand.scores.hook_strength < 6.8:
        flags.append("WEAK_HOOK")
        cand.review_notes.append(
            f"Quality Audit: Hook strength score ({cand.scores.hook_strength:.1f}/10) is below the 6.8 threshold."
        )

    if cand.scores.cliffhanger_strength < 6.8:
        flags.append("WEAK_CLIFFHANGER")
        cand.review_notes.append(
            f"Quality Audit: Cliffhanger score ({cand.scores.cliffhanger_strength:.1f}/10) is below the 6.8 threshold; ending may feel too resolved."
        )

    if cand.scores.standalone_context < 6.5:
        flags.append("MISSING_STANDALONE_CONTEXT")
        cand.review_notes.append(
            f"Quality Audit: Standalone context ({cand.scores.standalone_context:.1f}/10) is low; clip relies on mid-season backstory unfamiliar to new viewers."
        )

    # 3. Check duration & timestamp validation status
    if cand.total_duration_sec < config.min_promo_duration_sec or cand.total_duration_sec > config.max_promo_duration_sec:
        flags.append("INVALID_DURATION")

    if cand.validation_status == "VALID_SNAPPED":
        flags.append("TIMESTAMP_SNAPPED")
    elif cand.validation_status in ("NEEDS_REVIEW", "INVALID"):
        flags.append("INCORRECT_TIMESTAMPS")

    # 4. Overall suitability check
    avg_core = 0.3 * cand.scores.hook_strength + 0.3 * cand.scores.cliffhanger_strength + 0.2 * cand.scores.standalone_context + 0.1 * cand.scores.emotional_impact + 0.1 * cand.scores.narrative_coherence
    if avg_core < 6.8 or ("WEAK_HOOK" in flags and "MISSING_STANDALONE_CONTEXT" in flags):
        flags.append("NOT_SUITABLE_FOR_PROMOTION")
        cand.review_notes.append(
            "Quality Audit: Scene is transitional/procedural and not suitable for cold-audience acquisition without heavy editing."
        )

    cand.quality_flags = flags
    if flags and cand.validation_status == "VALID" and any(
        f in flags for f in ("WEAK_HOOK", "WEAK_CLIFFHANGER", "MISSING_STANDALONE_CONTEXT", "NOT_SUITABLE_FOR_PROMOTION", "WEAK_OPENING_3S_SLOW_CAMERA")
    ):
        cand.validation_status = "NEEDS_REVIEW"
        if cand.confidence_level == "HIGH":
            cand.confidence_level = "MEDIUM"

    return flags


def generate_quality_evaluation_report(
    ranked_candidates: List[PromoCandidate],
    duplicates: List[PromoCandidate],
    episode_summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Builds a comprehensive Quality Evaluation & Failure Mode Report required by Part 7 of the assignment,
    highlighting concrete examples where the tool picks well vs. where it picks badly.
    """
    flag_counts: Dict[str, int] = {}
    weak_examples: List[Dict[str, Any]] = []

    for cand in ranked_candidates:
        for fl in cand.quality_flags:
            flag_counts[fl] = flag_counts.get(fl, 0) + 1
        if cand.quality_flags or cand.final_score < 7.2:
            weak_examples.append(
                {
                    "promo_id": cand.promo_id,
                    "rank": cand.rank,
                    "episode_numbers": cand.episode_numbers,
                    "segments": [
                        f"Ep {s.episode_number} ({s.start_timestamp} - {s.end_timestamp})"
                        for s in cand.segments
                    ],
                    "total_duration_sec": cand.total_duration_sec,
                    "final_score": cand.final_score,
                    "scores": cand.scores.to_dict(),
                    "quality_flags": cand.quality_flags,
                    "why_it_failed_or_needs_review": " | ".join(cand.review_notes),
                    "hook_description": cand.hook_description,
                    "cliffhanger_description": cand.cliffhanger_description,
                }
            )

    # Sort weak examples from lowest score (worst pick) upward
    weak_examples.sort(key=lambda x: x["final_score"])

    valid_count = sum(1 for c in ranked_candidates if c.validation_status in ("VALID", "VALID_SNAPPED"))
    review_count = sum(1 for c in ranked_candidates if c.validation_status == "NEEDS_REVIEW")
    invalid_count = sum(1 for c in ranked_candidates if c.validation_status == "INVALID")

    human_ratings = {"Good": 0, "Average": 0, "Bad": 0, "Unrated": 0}
    for c in ranked_candidates:
        if c.human_rating in ("Good", "Average", "Bad"):
            human_ratings[c.human_rating] += 1
        else:
            human_ratings["Unrated"] += 1

    return {
        "total_episodes_analyzed": len(episode_summaries),
        "episodes_with_strong_material": sum(1 for e in episode_summaries if e.get("has_strong_promo_material")),
        "episodes_flagged_transitional": sum(1 for e in episode_summaries if not e.get("has_strong_promo_material")),
        "total_unique_candidates": len(ranked_candidates),
        "duplicates_removed": len(duplicates),
        "valid_promos_count": valid_count,
        "promos_requiring_review_count": review_count,
        "invalid_promos_count": invalid_count,
        "quality_flag_breakdown": flag_counts,
        "human_review_counts": human_ratings,
        "documented_poor_selections": weak_examples[:10],
    }


def save_human_rating(outputs_dir: str, promo_id: str, rating: str, reviewer_note: str = "") -> bool:
    """Persists a human reviewer rating ('Good', 'Average', 'Bad') back to outputs/promos.json."""
    promos_path = os.path.join(outputs_dir, "promos.json")
    if not os.path.exists(promos_path):
        return False
    with open(promos_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated = False
    for item in data.get("promos", []):
        if item.get("promo_id") == promo_id:
            item["human_rating"] = rating
            if reviewer_note:
                item.setdefault("review_notes", []).append(f"Human Review ({rating}): {reviewer_note}")
            updated = True
            break

    if updated:
        with open(promos_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    return updated
