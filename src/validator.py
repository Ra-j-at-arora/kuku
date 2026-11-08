"""
Deterministic Python Timestamp & Segment Validator for PromoAI.
Never blindly trusts AI-generated timestamps:
- Verifies episode existence
- Parses start/end timestamps
- Cross-checks against source screenplay exact block boundaries and Event IDs
- Snaps minor LLM drift (<= snap_tolerance_sec) to actual block boundaries and logs review notes
- Validates combined promo duration (20.0s to 60.0s)
- Extracts verbatim screenplay excerpts for human verification
"""
from typing import Dict, List, Tuple, Any, Optional
from src.config import AppConfig
from src.models import (
    EpisodeDocument,
    PromoSegment,
    PromoScores,
    PromoCandidate,
)
from src.parser import (
    parse_timestamp_to_seconds,
    format_seconds_to_timestamp,
    extract_script_excerpt,
)


def _find_nearest_boundary(target_sec: float, valid_map: Dict[str, float]) -> Tuple[Optional[str], Optional[float], float]:
    """Finds the closest valid timestamp in valid_map and returns (ts_str, ts_sec, abs_diff)."""
    if not valid_map:
        return None, None, 9999.0
    best_str, best_sec = min(valid_map.items(), key=lambda kv: abs(kv[1] - target_sec))
    return best_str, best_sec, round(abs(best_sec - target_sec), 3)


def validate_and_build_candidate(
    raw_cand: Dict[str, Any],
    episodes_by_num: Dict[int, EpisodeDocument],
    config: AppConfig,
    temp_id: str = "CAND_000",
) -> PromoCandidate:
    """
    Validates a raw candidate dictionary produced by Stage 1 against the ground-truth EpisodeDocuments.
    Populates exact segment durations, total promo duration, validation status, review notes, and verbatim excerpt.
    """
    review_notes: List[str] = []
    validated_segments: List[PromoSegment] = []
    episode_nums_ordered: List[int] = []
    excerpts: List[str] = []

    has_invalid = False
    has_needs_review = False
    has_snapped = False

    raw_segments = raw_cand.get("segments", [])
    if not raw_segments:
        has_invalid = True
        review_notes.append("Candidate contains zero segments.")

    for seg_idx, rseg in enumerate(raw_segments, start=1):
        ep_num = int(rseg.get("episode_number", raw_cand.get("_source_episode", -1)))
        s_str_raw = str(rseg.get("start_timestamp", "")).strip()
        e_str_raw = str(rseg.get("end_timestamp", "")).strip()
        s_ev_id = rseg.get("start_event_id")
        e_ev_id = rseg.get("end_event_id")
        seg_summary = str(rseg.get("segment_summary", "")).strip()

        if ep_num not in episodes_by_num:
            has_invalid = True
            review_notes.append(f"Segment {seg_idx}: Referenced Episode {ep_num} does not exist in screenplay pack.")
            validated_segments.append(
                PromoSegment(
                    episode_number=ep_num,
                    start_timestamp=s_str_raw,
                    end_timestamp=e_str_raw,
                    validation_note="INVALID_EPISODE",
                )
            )
            continue

        if ep_num not in episode_nums_ordered:
            episode_nums_ordered.append(ep_num)

        ep_doc = episodes_by_num[ep_num]
        ev_by_id = {ev.event_id: ev for ev in ep_doc.events}
        valid_starts = ep_doc.get_valid_start_map()
        valid_ends = ep_doc.get_valid_end_map()

        # Parse start and end timestamps
        s_sec: Optional[float] = None
        e_sec: Optional[float] = None
        try:
            s_sec = parse_timestamp_to_seconds(s_str_raw)
        except ValueError:
            # Recover from start_event_id if possible
            if s_ev_id and s_ev_id in ev_by_id:
                s_sec = ev_by_id[s_ev_id].start_sec
                s_str_raw = ev_by_id[s_ev_id].start_str
                has_snapped = True
                review_notes.append(
                    f"Segment {seg_idx} (Ep {ep_num}): Recovered malformed start timestamp from event ID {s_ev_id} ({s_str_raw})."
                )
            else:
                has_invalid = True
                review_notes.append(f"Segment {seg_idx} (Ep {ep_num}): Malformed start timestamp {s_str_raw!r}.")

        try:
            e_sec = parse_timestamp_to_seconds(e_str_raw)
        except ValueError:
            if e_ev_id and e_ev_id in ev_by_id:
                e_sec = ev_by_id[e_ev_id].end_sec
                e_str_raw = ev_by_id[e_ev_id].end_str
                has_snapped = True
                review_notes.append(
                    f"Segment {seg_idx} (Ep {ep_num}): Recovered malformed end timestamp from event ID {e_ev_id} ({e_str_raw})."
                )
            else:
                has_invalid = True
                review_notes.append(f"Segment {seg_idx} (Ep {ep_num}): Malformed end timestamp {e_str_raw!r}.")

        if s_sec is None or e_sec is None:
            validated_segments.append(
                PromoSegment(
                    episode_number=ep_num,
                    start_timestamp=s_str_raw,
                    end_timestamp=e_str_raw,
                    start_event_id=s_ev_id,
                    end_event_id=e_ev_id,
                    segment_summary=seg_summary,
                    validation_note="INVALID_TIMESTAMP_FORMAT",
                )
            )
            continue

        seg_notes = []

        # Verify start timestamp against screenplay boundaries
        if s_str_raw in valid_starts:
            s_sec = valid_starts[s_str_raw]
            seg_notes.append("EXACT_START")
        else:
            near_s_str, near_s_sec, diff_s = _find_nearest_boundary(s_sec, valid_starts)
            if near_s_str is not None and near_s_sec is not None and diff_s <= config.snap_tolerance_sec:
                review_notes.append(
                    f"Segment {seg_idx} (Ep {ep_num}): Start '{s_str_raw}' snapped by {diff_s:.2f}s to exact block start '{near_s_str}'."
                )
                s_str_raw = near_s_str
                s_sec = near_s_sec
                has_snapped = True
                seg_notes.append(f"SNAPPED_START({diff_s:.2f}s)")
            elif s_ev_id and s_ev_id in ev_by_id:
                s_str_raw = ev_by_id[s_ev_id].start_str
                s_sec = ev_by_id[s_ev_id].start_sec
                has_snapped = True
                review_notes.append(
                    f"Segment {seg_idx} (Ep {ep_num}): Start aligned via Event ID {s_ev_id} to '{s_str_raw}'."
                )
                seg_notes.append("EVENT_ID_ALIGNED_START")
            else:
                has_needs_review = True
                review_notes.append(
                    f"Segment {seg_idx} (Ep {ep_num}): Start '{s_str_raw}' does not match any block boundary (nearest is {near_s_str}, off by {diff_s:.2f}s)."
                )
                seg_notes.append("UNVERIFIED_START")

        # Verify end timestamp against screenplay boundaries
        if e_str_raw in valid_ends:
            e_sec = valid_ends[e_str_raw]
            seg_notes.append("EXACT_END")
        else:
            near_e_str, near_e_sec, diff_e = _find_nearest_boundary(e_sec, valid_ends)
            if near_e_str is not None and near_e_sec is not None and diff_e <= config.snap_tolerance_sec:
                review_notes.append(
                    f"Segment {seg_idx} (Ep {ep_num}): End '{e_str_raw}' snapped by {diff_e:.2f}s to exact block end '{near_e_str}'."
                )
                e_str_raw = near_e_str
                e_sec = near_e_sec
                has_snapped = True
                seg_notes.append(f"SNAPPED_END({diff_e:.2f}s)")
            elif e_ev_id and e_ev_id in ev_by_id:
                e_str_raw = ev_by_id[e_ev_id].end_str
                e_sec = ev_by_id[e_ev_id].end_sec
                has_snapped = True
                review_notes.append(
                    f"Segment {seg_idx} (Ep {ep_num}): End aligned via Event ID {e_ev_id} to '{e_str_raw}'."
                )
                seg_notes.append("EVENT_ID_ALIGNED_END")
            else:
                has_needs_review = True
                review_notes.append(
                    f"Segment {seg_idx} (Ep {ep_num}): End '{e_str_raw}' does not match any block boundary (nearest is {near_e_str}, off by {diff_e:.2f}s)."
                )
                seg_notes.append("UNVERIFIED_END")

        # Check ordering and episode bounds
        if e_sec <= s_sec:
            has_invalid = True
            review_notes.append(
                f"Segment {seg_idx} (Ep {ep_num}): End timestamp ({e_str_raw}) <= Start timestamp ({s_str_raw})."
            )
            seg_notes.append("INVALID_ORDER")

        if e_sec > ep_doc.total_duration_sec + 1.0:
            has_invalid = True
            review_notes.append(
                f"Segment {seg_idx} (Ep {ep_num}): End timestamp ({e_str_raw}) exceeds episode duration ({ep_doc.total_duration_str})."
            )
            seg_notes.append("OUT_OF_BOUNDS")

        seg_dur = round(max(0.0, e_sec - s_sec), 2)
        validated_segments.append(
            PromoSegment(
                episode_number=ep_num,
                start_timestamp=s_str_raw,
                end_timestamp=e_str_raw,
                start_sec=round(s_sec, 2),
                end_sec=round(e_sec, 2),
                duration_sec=seg_dur,
                start_event_id=s_ev_id,
                end_event_id=e_ev_id,
                segment_summary=seg_summary,
                validation_note=" | ".join(seg_notes),
            )
        )

        if not has_invalid and e_sec > s_sec:
            excerpt_text = extract_script_excerpt(ep_doc, s_sec, e_sec)
            excerpts.append(f"=== Episode {ep_num} ({s_str_raw} - {e_str_raw}) ===\n{excerpt_text}")

    total_duration = round(sum(seg.duration_sec for seg in validated_segments), 2)

    # Check total duration constraints [20.0s, 60.0s]
    if total_duration < config.min_promo_duration_sec:
        has_invalid = True
        review_notes.append(
            f"Combined duration ({total_duration:.2f}s) is below minimum {config.min_promo_duration_sec:.1f}s requirement."
        )
    elif total_duration > config.max_promo_duration_sec:
        has_invalid = True
        review_notes.append(
            f"Combined duration ({total_duration:.2f}s) exceeds maximum {config.max_promo_duration_sec:.1f}s requirement."
        )

    # Determine overall validation_status
    if has_invalid:
        val_status = "INVALID"
    elif has_needs_review:
        val_status = "NEEDS_REVIEW"
    elif has_snapped:
        val_status = "VALID_SNAPPED"
    else:
        val_status = "VALID"

    if not review_notes:
        review_notes.append("All timestamps verified against exact screenplay block boundaries; duration within 20-60s.")

    raw_scores = raw_cand.get("scores", {})
    scores_obj = PromoScores(
        hook_strength=float(raw_scores.get("hook_strength", 5.0)),
        cliffhanger_strength=float(raw_scores.get("cliffhanger_strength", 5.0)),
        standalone_context=float(raw_scores.get("standalone_context", 5.0)),
        emotional_impact=float(raw_scores.get("emotional_impact", 5.0)),
        narrative_coherence=float(raw_scores.get("narrative_coherence", 5.0)),
    )

    # Determine promo_type
    if len(episode_nums_ordered) > 1:
        promo_type = "CROSS_EPISODE"
    elif len(validated_segments) > 1:
        promo_type = "MULTI_SEGMENT_SAME_EP"
    else:
        promo_type = "SINGLE_SEGMENT"

    return PromoCandidate(
        promo_id=temp_id,
        rank=0,
        episode_numbers=episode_nums_ordered,
        promo_type=promo_type,
        segments=validated_segments,
        total_duration_sec=total_duration,
        hook_description=str(raw_cand.get("hook_description", "")).strip(),
        cliffhanger_description=str(raw_cand.get("cliffhanger_description", "")).strip(),
        standalone_context=str(raw_cand.get("standalone_context", "")).strip(),
        why_it_works=str(raw_cand.get("why_it_works", "")).strip(),
        scores=scores_obj,
        final_score=0.0,
        confidence_level=str(raw_cand.get("confidence_level", "MEDIUM")).upper(),
        validation_status=val_status,
        review_notes=review_notes,
        quality_flags=[],
        human_rating=None,
        script_excerpt="\n\n".join(excerpts),
    )
