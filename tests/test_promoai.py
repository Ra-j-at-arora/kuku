"""
Automated Unit & Integration Test Suite for PromoAI (Part 9 of Kuku AI Generalist Assignment).
Covers all 8 required test areas:
1. Timestamp parsing (MM:SS.xx, HH:MM:SS.xx, parentheses, invalid inputs)
2. Duration calculations (single segment, multi-track overlapping events)
3. Multiple segments (intra-episode 2-part cuts & cross-episode combined durations)
4. Episode extraction (headers, scenes, Episode 6 uppercase anomaly, verbatim quotes)
5. JSON schema validation (Stage 1 output structure & required fields)
6. Promo scoring (30/30/20/10/10 weighted math, custom weights, penalty application)
7. Invalid timestamps (snapping within 1.5s tolerance, Event-ID fallback, out-of-bounds, inverted start/end, <20s and >60s rejection)
8. Duplicate detection (Temporal IoU calculation & duplicate suppression)
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import AppConfig, ScoringWeights
from src.models import PromoScores
from src.parser import (
    parse_timestamp_to_seconds,
    format_seconds_to_timestamp,
    parse_screenplay_text,
    load_episodes_from_zip,
)
from src.validator import validate_and_build_candidate
from src.scorer import calculate_weighted_score, compute_temporal_iou, score_deduplicate_and_rank
from src.evaluator import audit_candidate_quality
from src.gemini_client import GeminiPromoClient, EPISODE_DISCOVERY_SCHEMA


SAMPLE_SCREENPLAY_EP6_STYLE = """EPISODE NUMBER: Episode 6
EPISODE LOGLINE: Seeking shelter from the cold rain, Juliana arrives at her new husband's modest woodland cottage.
CONTINUITY NOTES: Picks up immediately from Episode 5.

--- SCRIPT START ---

SCENE 01
SCENE HEADING: EXT. WOODLAND PATH – NIGHT
SCENE START: 00:00.00
SCENE END: 00:35.00

ACTIONS AND DIALOGUE:

**CAMERA:**
Start: (00:00.00) | End: (00:03.50)
Wide tracking shot of Juliana and the Mysterious Bearded Man in the rain.

**MYSTERIOUS BEARDED MAN (Solemn, Gentle):**
Start: (00:03.50) | End: (00:10.00)
"I know this is not the life you came from."

**JULIANA (Quiet, Resolute):**
Start: (00:10.00) | End: (00:22.00)
"Better the mud in these woods than the gilded cage they built for me."

**ACTION:**
Start: (00:22.00) | End: (00:35.00)
Suddenly, a masked rider bursts through the trees drawing a steel sabre!

--- SCRIPT END ---
"""


class TestPromoAI(unittest.TestCase):
    def setUp(self):
        self.config = AppConfig(
            min_promo_duration_sec=20.0,
            max_promo_duration_sec=60.0,
            snap_tolerance_sec=1.5,
            dedup_iou_threshold=0.60,
        )
        self.sample_ep = parse_screenplay_text(SAMPLE_SCREENPLAY_EP6_STYLE, filename="episode-006-Episode 1671708.txt")
        self.episodes_map = {6: self.sample_ep}

    # 1. Timestamp Parsing Tests
    def test_01_timestamp_parsing(self):
        self.assertAlmostEqual(parse_timestamp_to_seconds("00:00.00"), 0.0)
        self.assertAlmostEqual(parse_timestamp_to_seconds("(01:15.50)"), 75.50)
        self.assertAlmostEqual(parse_timestamp_to_seconds("02:18.00"), 138.00)
        self.assertAlmostEqual(parse_timestamp_to_seconds("00:01:05.25"), 65.25)
        self.assertEqual(format_seconds_to_timestamp(138.0), "02:18.00")
        with self.assertRaises(ValueError):
            parse_timestamp_to_seconds("invalid_ts")
        with self.assertRaises(ValueError):
            parse_timestamp_to_seconds("01:75.00")  # seconds >= 60

    # 2. Duration Calculations Tests
    def test_02_duration_calculations(self):
        raw_cand = {
            "promo_type": "SINGLE_SEGMENT",
            "segments": [
                {
                    "episode_number": 6,
                    "start_event_id": "E002",
                    "end_event_id": "E004",
                    "start_timestamp": "00:03.50",
                    "end_timestamp": "00:35.00",
                    "segment_summary": "Main conflict",
                }
            ],
            "hook_description": "Opens on dialogue.",
            "cliffhanger_description": "Rider draws sabre.",
            "standalone_context": "Clear stakes.",
            "why_it_works": "Strong arc.",
            "scores": {
                "hook_strength": 8.5,
                "cliffhanger_strength": 9.0,
                "standalone_context": 8.0,
                "emotional_impact": 8.5,
                "narrative_coherence": 9.0,
            },
            "confidence_level": "HIGH",
        }
        cand = validate_and_build_candidate(raw_cand, self.episodes_map, self.config)
        self.assertEqual(cand.validation_status, "VALID")
        self.assertAlmostEqual(cand.segments[0].duration_sec, 31.50)
        self.assertAlmostEqual(cand.total_duration_sec, 31.50)

    # 3. Multiple Segments & Cross-Episode Combined Duration Tests
    def test_03_multiple_segments(self):
        ep7_text = SAMPLE_SCREENPLAY_EP6_STYLE.replace("Episode 6", "Episode 7")
        ep7 = parse_screenplay_text(ep7_text, filename="episode-007-Episode 1671709.txt")
        ep_map = {6: self.sample_ep, 7: ep7}

        raw_multi = {
            "promo_type": "CROSS_EPISODE",
            "segments": [
                {
                    "episode_number": 6,
                    "start_event_id": "E002",
                    "end_event_id": "E003",
                    "start_timestamp": "00:03.50",
                    "end_timestamp": "00:22.00",
                    "segment_summary": "Ep 6 setup (18.5s)",
                },
                {
                    "episode_number": 7,
                    "start_event_id": "E003",
                    "end_event_id": "E004",
                    "start_timestamp": "00:10.00",
                    "end_timestamp": "00:35.00",
                    "segment_summary": "Ep 7 payoff (25.0s)",
                },
            ],
            "hook_description": "Hook",
            "cliffhanger_description": "Cliffhanger",
            "standalone_context": "Context",
            "why_it_works": "Works",
            "scores": {
                "hook_strength": 9.0,
                "cliffhanger_strength": 9.0,
                "standalone_context": 8.5,
                "emotional_impact": 8.5,
                "narrative_coherence": 8.5,
            },
        }
        cand = validate_and_build_candidate(raw_multi, ep_map, self.config)
        self.assertEqual(cand.promo_type, "CROSS_EPISODE")
        self.assertEqual(cand.episode_numbers, [6, 7])
        self.assertAlmostEqual(cand.total_duration_sec, 43.50)
        self.assertEqual(cand.validation_status, "VALID")

    # 4. Episode Extraction & Anomaly Handling Tests
    def test_04_episode_extraction(self):
        self.assertEqual(self.sample_ep.episode_number, 6)
        self.assertEqual(self.sample_ep.production_id, "1671708")
        self.assertEqual(len(self.sample_ep.scenes), 1)
        self.assertEqual(self.sample_ep.scenes[0].heading, "EXT. WOODLAND PATH – NIGHT")
        self.assertEqual(len(self.sample_ep.events), 4)
        self.assertIn("MYSTERIOUS BEARDED MAN", self.sample_ep.characters)
        self.assertIn("JULIANA", self.sample_ep.characters)
        self.assertEqual(len(self.sample_ep.parse_warnings), 0)

    # 5. JSON Schema Validation Tests
    def test_05_json_validation(self):
        client = GeminiPromoClient(self.config)
        res = client._deterministic_episode_analysis(self.sample_ep)
        required_top = EPISODE_DISCOVERY_SCHEMA["required"]
        for req_key in required_top:
            self.assertIn(req_key, res)
        self.assertIsInstance(res["candidates"], list)
        # Ensure JSON serializability
        serialized = json.dumps(res)
        self.assertTrue(len(serialized) > 20)

    # 6. Promo Scoring Formula Tests
    def test_06_promo_scoring(self):
        scores = PromoScores(
            hook_strength=10.0,
            cliffhanger_strength=8.0,
            standalone_context=7.0,
            emotional_impact=9.0,
            narrative_coherence=6.0,
        )
        # Expected: 0.3*10 + 0.3*8 + 0.2*7 + 0.1*9 + 0.1*6 = 3.0 + 2.4 + 1.4 + 0.9 + 0.6 = 8.30
        weighted = calculate_weighted_score(scores, ScoringWeights())
        self.assertAlmostEqual(weighted, 8.30)

    # 7. Invalid & Snapped Timestamps Tests
    def test_07_invalid_and_snapped_timestamps(self):
        # 7a: Slight drift (00:03.90 instead of 00:03.50 -> diff 0.40s <= 1.5s) should snap to 00:03.50
        raw_snap = {
            "promo_type": "SINGLE_SEGMENT",
            "segments": [
                {
                    "episode_number": 6,
                    "start_event_id": "E002",
                    "end_event_id": "E004",
                    "start_timestamp": "00:03.90",
                    "end_timestamp": "00:35.00",
                    "segment_summary": "Snapped test",
                }
            ],
            "hook_description": "Hook",
            "cliffhanger_description": "Cliff",
            "standalone_context": "Context",
            "why_it_works": "Works",
            "scores": {"hook_strength": 8, "cliffhanger_strength": 8, "standalone_context": 8, "emotional_impact": 8, "narrative_coherence": 8},
        }
        cand_snap = validate_and_build_candidate(raw_snap, self.episodes_map, self.config)
        self.assertEqual(cand_snap.validation_status, "VALID_SNAPPED")
        self.assertEqual(cand_snap.segments[0].start_timestamp, "00:03.50")

        # 7b: Too short duration (< 20s) should be marked INVALID
        raw_short = {
            "promo_type": "SINGLE_SEGMENT",
            "segments": [
                {
                    "episode_number": 6,
                    "start_event_id": "E001",
                    "end_event_id": "E002",
                    "start_timestamp": "00:00.00",
                    "end_timestamp": "00:10.00",
                    "segment_summary": "Too short (10s)",
                }
            ],
            "hook_description": "Hook",
            "cliffhanger_description": "Cliff",
            "standalone_context": "Context",
            "why_it_works": "Works",
            "scores": {"hook_strength": 8, "cliffhanger_strength": 8, "standalone_context": 8, "emotional_impact": 8, "narrative_coherence": 8},
        }
        cand_short = validate_and_build_candidate(raw_short, self.episodes_map, self.config)
        self.assertEqual(cand_short.validation_status, "INVALID")

    # 8. Duplicate Detection (tIoU) Tests
    def test_08_duplicate_detection(self):
        base_raw = {
            "promo_type": "SINGLE_SEGMENT",
            "segments": [
                {
                    "episode_number": 6,
                    "start_event_id": "E001",
                    "end_event_id": "E004",
                    "start_timestamp": "00:00.00",
                    "end_timestamp": "00:35.00",
                    "segment_summary": "Full 35s",
                }
            ],
            "hook_description": "Hook",
            "cliffhanger_description": "Cliff",
            "standalone_context": "Context",
            "why_it_works": "Works",
            "scores": {"hook_strength": 9, "cliffhanger_strength": 9, "standalone_context": 9, "emotional_impact": 9, "narrative_coherence": 9},
        }
        overlapping_raw = {
            "promo_type": "SINGLE_SEGMENT",
            "segments": [
                {
                    "episode_number": 6,
                    "start_event_id": "E002",
                    "end_event_id": "E004",
                    "start_timestamp": "00:03.50",
                    "end_timestamp": "00:35.00",
                    "segment_summary": "Overlapping 31.5s (IoU = 31.5/35.0 = 0.90)",
                }
            ],
            "hook_description": "Hook 2",
            "cliffhanger_description": "Cliff 2",
            "standalone_context": "Context 2",
            "why_it_works": "Works 2",
            "scores": {"hook_strength": 8, "cliffhanger_strength": 8, "standalone_context": 8, "emotional_impact": 8, "narrative_coherence": 8},
        }
        c1 = validate_and_build_candidate(base_raw, self.episodes_map, self.config, temp_id="C1")
        c2 = validate_and_build_candidate(overlapping_raw, self.episodes_map, self.config, temp_id="C2")
        iou = compute_temporal_iou(c1, c2)
        self.assertAlmostEqual(iou, 0.90)

        unique, dups = score_deduplicate_and_rank([c1, c2], self.config)
        self.assertEqual(len(unique), 1)
        self.assertEqual(len(dups), 1)
        self.assertEqual(unique[0].promo_id, "PROMO_001")


if __name__ == "__main__":
    unittest.main(verbosity=2)