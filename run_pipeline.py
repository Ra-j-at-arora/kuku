"""
End-to-End Pipeline Orchestrator and Exporter for PromoAI.
Can be invoked from CLI (`python3 run_pipeline.py --zip path/to/scripts.zip`)
or imported directly by the Streamlit frontend (`app.py`).
"""
import argparse
import csv
import json
import os
import time
from typing import List, Dict, Any, Optional, Callable

from src.config import AppConfig
from src.models import EpisodeDocument, PromoCandidate
from src.parser import load_episodes_from_zip, load_episodes_from_directory
from src.discovery import DiscoveryPipeline
from src.validator import validate_and_build_candidate
from src.scorer import score_deduplicate_and_rank
from src.evaluator import audit_candidate_quality, generate_quality_evaluation_report


def export_results(
    ranked_candidates: List[PromoCandidate],
    duplicates: List[PromoCandidate],
    episode_summaries: List[Dict[str, Any]],
    evaluation_report: Dict[str, Any],
    processing_stats: Dict[str, Any],
    outputs_dir: str,
) -> Dict[str, str]:
    """
    Exports all required deliverables to `outputs/`:
    - outputs/promos.json
    - outputs/promos.csv
    - outputs/episode_analysis.json
    - outputs/evaluation_report.json
    """
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "raw_model_responses"), exist_ok=True)

    promos_json_path = os.path.join(outputs_dir, "promos.json")
    promos_csv_path = os.path.join(outputs_dir, "promos.csv")
    ep_analysis_path = os.path.join(outputs_dir, "episode_analysis.json")
    eval_report_path = os.path.join(outputs_dir, "evaluation_report.json")

    promos_payload = {
        "project": "PromoAI - AI-Powered Microdrama Promo Finder",
        "processing_stats": processing_stats,
        "promos": [c.to_dict() for c in ranked_candidates],
        "duplicates_archived": [d.to_dict() for d in duplicates],
    }
    with open(promos_json_path, "w", encoding="utf-8") as f:
        json.dump(promos_payload, f, indent=2, ensure_ascii=False)

    # Export flat CSV for editors & growth marketers
    csv_columns = [
        "promo_id",
        "rank",
        "promo_type",
        "episode_numbers",
        "segments_formatted",
        "total_duration_sec",
        "final_score",
        "hook_strength",
        "cliffhanger_strength",
        "standalone_context",
        "emotional_impact",
        "narrative_coherence",
        "confidence_level",
        "validation_status",
        "quality_flags",
        "human_rating",
        "hook_description",
        "cliffhanger_description",
        "standalone_context_explanation",
        "why_it_works",
        "review_notes",
    ]
    with open(promos_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for c in ranked_candidates:
            seg_str = " ; ".join(
                f"Ep {s.episode_number}: {s.start_timestamp}-{s.end_timestamp} ({s.duration_sec:.2f}s)"
                for s in c.segments
            )
            writer.writerow(
                {
                    "promo_id": c.promo_id,
                    "rank": c.rank,
                    "promo_type": c.promo_type,
                    "episode_numbers": ",".join(str(x) for x in c.episode_numbers),
                    "segments_formatted": seg_str,
                    "total_duration_sec": c.total_duration_sec,
                    "final_score": c.final_score,
                    "hook_strength": c.scores.hook_strength,
                    "cliffhanger_strength": c.scores.cliffhanger_strength,
                    "standalone_context": c.scores.standalone_context,
                    "emotional_impact": c.scores.emotional_impact,
                    "narrative_coherence": c.scores.narrative_coherence,
                    "confidence_level": c.confidence_level,
                    "validation_status": c.validation_status,
                    "quality_flags": "|".join(c.quality_flags),
                    "human_rating": c.human_rating or "",
                    "hook_description": c.hook_description,
                    "cliffhanger_description": c.cliffhanger_description,
                    "standalone_context_explanation": c.standalone_context,
                    "why_it_works": c.why_it_works,
                    "review_notes": " | ".join(c.review_notes),
                }
            )

    with open(ep_analysis_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_episodes": len(episode_summaries),
                "episodes": episode_summaries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(eval_report_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2, ensure_ascii=False)

    return {
        "promos_json": promos_json_path,
        "promos_csv": promos_csv_path,
        "episode_analysis_json": ep_analysis_path,
        "evaluation_report_json": eval_report_path,
    }


def execute_full_pipeline(
    episodes: List[EpisodeDocument],
    config: Optional[AppConfig] = None,
    use_cache: bool = True,
    progress_callback: Optional[Callable[[int, int, EpisodeDocument, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Runs Stage 1 Discovery, Stage 2 Validation & Scoring, and Stage 3 Quality Evaluation."""
    cfg = config or AppConfig()
    t0 = time.time()

    episodes_by_num = {ep.episode_number: ep for ep in episodes}
    pipeline = DiscoveryPipeline(cfg)

    raw_candidates, episode_summaries = pipeline.run_episode_discovery(
        episodes=episodes,
        use_cache=use_cache,
        progress_callback=progress_callback,
    )

    validated_candidates: List[PromoCandidate] = []
    for idx, rc in enumerate(raw_candidates, start=1):
        cand = validate_and_build_candidate(
            raw_cand=rc,
            episodes_by_num=episodes_by_num,
            config=cfg,
            temp_id=f"CAND_{idx:03d}",
        )
        audit_candidate_quality(cand, episodes_by_num, cfg)
        validated_candidates.append(cand)

    ranked_candidates, duplicates = score_deduplicate_and_rank(validated_candidates, cfg)
    eval_report = generate_quality_evaluation_report(ranked_candidates, duplicates, episode_summaries)

    elapsed = round(time.time() - t0, 2)
    stats = {
        "total_episodes_input": len(episodes),
        "successfully_processed_episodes": len(episode_summaries),
        "failed_episodes": len(episodes) - len(episode_summaries),
        "total_raw_candidates_discovered": len(raw_candidates),
        "unique_ranked_promos": len(ranked_candidates),
        "duplicates_removed": len(duplicates),
        "valid_promos": sum(1 for c in ranked_candidates if c.validation_status in ("VALID", "VALID_SNAPPED")),
        "promos_requiring_review": sum(1 for c in ranked_candidates if c.validation_status == "NEEDS_REVIEW"),
        "invalid_promos": sum(1 for c in ranked_candidates if c.validation_status == "INVALID"),
        "single_segment_promos": sum(1 for c in ranked_candidates if c.promo_type == "SINGLE_SEGMENT"),
        "multi_segment_same_ep_promos": sum(1 for c in ranked_candidates if c.promo_type == "MULTI_SEGMENT_SAME_EP"),
        "cross_episode_promos": sum(1 for c in ranked_candidates if c.promo_type == "CROSS_EPISODE"),
        "processing_time_sec": elapsed,
        "model_configured": cfg.gemini_model,
        "engine_mode": pipeline.client.active_mode,
        "weights_used": cfg.weights.normalized(),
    }

    paths = export_results(
        ranked_candidates=ranked_candidates,
        duplicates=duplicates,
        episode_summaries=episode_summaries,
        evaluation_report=eval_report,
        processing_stats=stats,
        outputs_dir=cfg.outputs_dir,
    )

    return {
        "stats": stats,
        "promos": ranked_candidates,
        "duplicates": duplicates,
        "episode_summaries": episode_summaries,
        "evaluation_report": eval_report,
        "export_paths": paths,
    }


def main():
    parser = argparse.ArgumentParser(description="Run PromoAI Pipeline on a screenplay ZIP or folder.")
    parser.add_argument("--zip", type=str, help="Path to screenplay ZIP archive.")
    parser.add_argument("--dir", type=str, help="Path to directory containing screenplay .txt files.")
    parser.add_argument("--outputs", type=str, default="outputs", help="Directory for output artifacts.")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached responses in raw_model_responses.")
    args = parser.parse_args()

    cfg = AppConfig(outputs_dir=args.outputs)
    if args.zip:
        episodes = load_episodes_from_zip(args.zip)
    elif args.dir:
        episodes = load_episodes_from_directory(args.dir)
    else:
        raise SystemExit("Please specify --zip <path.zip> or --dir <folder>.")

    print(f"[PromoAI] Loaded {len(episodes)} episodes. Starting two-stage pipeline...")
    res = execute_full_pipeline(
        episodes,
        config=cfg,
        use_cache=not args.no_cache,
        progress_callback=lambda i, n, ep, _: print(f"  [{i:02d}/{n:02d}] Processed Episode {ep.episode_number} ({ep.total_duration_str})"),
    )
    st = res["stats"]
    print("\n=== PromoAI Run Summary ===")
    for k, v in st.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
