"""
Stage 2 Candidate Evaluation, Configurable Scoring, Deduplication, and Ranking in Python.
Formula:
  Hook strength:        30% (configurable)
  Cliffhanger strength: 30% (configurable)
  Standalone context:   20% (configurable)
  Emotional impact:     10% (configurable)
  Narrative coherence:  10% (configurable)
Each sub-score is 0-10. Python computes both the 0-10 weighted average and 0-100 scaled score.
"""
from typing import List, Tuple
from src.config import AppConfig, ScoringWeights
from src.models import PromoCandidate, PromoScores


def calculate_weighted_score(scores: PromoScores, weights: ScoringWeights) -> float:
    """
    Calculates the deterministic weighted score on a 0-10 scale (rounded to 2 decimals)
    from individual 0-10 parameter scores.
    """
    w = weights.normalized()
    h = max(0.0, min(10.0, float(scores.hook_strength)))
    c = max(0.0, min(10.0, float(scores.cliffhanger_strength)))
    s = max(0.0, min(10.0, float(scores.standalone_context)))
    e = max(0.0, min(10.0, float(scores.emotional_impact)))
    n = max(0.0, min(10.0, float(scores.narrative_coherence)))

    weighted = (
        w["hook_strength"] * h
        + w["cliffhanger_strength"] * c
        + w["standalone_context"] * s
        + w["emotional_impact"] * e
        + w["narrative_coherence"] * n
    )
    return round(weighted, 2)


def compute_temporal_iou(cand_a: PromoCandidate, cand_b: PromoCandidate) -> float:
    """
    Computes Temporal Intersection-over-Union (tIoU) between two PromoCandidates.
    Returns 0.0 if they do not share identical episode_numbers.
    """
    if cand_a.episode_numbers != cand_b.episode_numbers:
        return 0.0
    if len(cand_a.segments) != len(cand_b.segments):
        return 0.0

    intersection = 0.0
    union = 0.0
    for seg_a, seg_b in zip(cand_a.segments, cand_b.segments):
        if seg_a.episode_number != seg_b.episode_number:
            return 0.0
        inter_start = max(seg_a.start_sec, seg_b.start_sec)
        inter_end = min(seg_a.end_sec, seg_b.end_sec)
        inter = max(0.0, inter_end - inter_start)
        uni = max(seg_a.end_sec, seg_b.end_sec) - min(seg_a.start_sec, seg_b.start_sec)
        intersection += inter
        union += max(0.001, uni)

    return round(intersection / max(0.001, union), 3)


def score_deduplicate_and_rank(
    candidates: List[PromoCandidate],
    config: AppConfig,
) -> Tuple[List[PromoCandidate], List[PromoCandidate]]:
    """
    1. Computes final weighted score for every candidate in Python.
    2. Applies validation penalty to INVALID / NEEDS_REVIEW candidates so valid candidates rank first.
    3. Deduplicates overlapping candidates (tIoU >= config.dedup_iou_threshold).
    4. Assigns global rank (1..N) and sequential Promo IDs (PROMO_001..PROMO_NNN).
    Returns (ranked_unique_candidates, removed_duplicates).
    """
    for cand in candidates:
        base_score = calculate_weighted_score(cand.scores, config.weights)
        if cand.validation_status == "INVALID":
            cand.final_score = round(max(0.0, base_score - 3.5), 2)
            cand.confidence_level = "LOW"
        elif cand.validation_status == "NEEDS_REVIEW":
            cand.final_score = round(max(0.0, base_score - 1.0), 2)
            if cand.confidence_level == "HIGH":
                cand.confidence_level = "MEDIUM"
        else:
            cand.final_score = base_score

    # Sort descending by (is_valid, final_score, hook_strength, cliffhanger_strength)
    sorted_all = sorted(
        candidates,
        key=lambda c: (
            1 if c.validation_status in ("VALID", "VALID_SNAPPED") else 0,
            c.final_score,
            c.scores.hook_strength,
            c.scores.cliffhanger_strength,
        ),
        reverse=True,
    )

    unique_candidates: List[PromoCandidate] = []
    duplicates: List[PromoCandidate] = []

    for cand in sorted_all:
        is_dup = False
        for kept in unique_candidates:
            iou = compute_temporal_iou(cand, kept)
            if iou >= config.dedup_iou_threshold:
                cand.quality_flags.append(f"DUPLICATE_OVERLAP(IoU={iou:.2f} with {kept.promo_id})")
                cand.review_notes.append(
                    f"Flagged as duplicate overlap (IoU={iou:.2f}) with higher-ranked candidate on Episode(s) {kept.episode_numbers}."
                )
                duplicates.append(cand)
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(cand)

    for rank_idx, cand in enumerate(unique_candidates, start=1):
        cand.rank = rank_idx
        cand.promo_id = f"PROMO_{rank_idx:03d}"

    for dup_idx, dup in enumerate(duplicates, start=1):
        dup.rank = len(unique_candidates) + dup_idx
        dup.promo_id = f"DUP_{dup_idx:03d}"

    return unique_candidates, duplicates
