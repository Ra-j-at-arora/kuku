"""Data models for PromoAI screenplay representation, candidates, validation, and ranking."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class ScriptEvent:
    """Represents a single timed event block (CAMERA, ACTION, SOUND_FX, or DIALOGUE) in a screenplay."""
    event_id: str                 # e.g., "E001"
    scene_number: int             # e.g., 1
    event_type: str               # "CAMERA", "ACTION", "SOUND_FX", or "DIALOGUE"
    raw_header: str               # Verbatim bold header without asterisks, e.g., "MATRIARCH (Cold, Commanding)"
    speaker: Optional[str]        # Speaker name if DIALOGUE, e.g., "MATRIARCH"
    delivery_modifier: Optional[str]  # Parenthetical emotion/direction, e.g., "Cold, Commanding"
    start_str: str                # Verbatim start string, e.g., "00:11.30"
    end_str: str                  # Verbatim end string, e.g., "00:13.10"
    start_sec: float              # Parsed float seconds, e.g., 11.30
    end_sec: float                # Parsed float seconds, e.g., 13.10
    duration_sec: float           # end_sec - start_sec
    text: str                     # Verbatim text content (dialogue keeps original quotes)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SceneBlock:
    """Represents a scene within an episode screenplay."""
    scene_number: int
    heading: str
    start_str: str
    end_str: str
    start_sec: float
    end_sec: float
    events: List[ScriptEvent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeDocument:
    """Structured representation of a full episode screenplay preserving 100% of original content."""
    episode_number: int
    production_id: str
    source_filename: str
    logline: str
    continuity_notes: str
    total_duration_sec: float
    total_duration_str: str
    scenes: List[SceneBlock] = field(default_factory=list)
    events: List[ScriptEvent] = field(default_factory=list)
    characters: List[str] = field(default_factory=list)
    parse_warnings: List[str] = field(default_factory=list)

    def get_valid_start_map(self) -> Dict[str, float]:
        """Returns mapping of exact start timestamp strings to seconds."""
        mp = {ev.start_str: ev.start_sec for ev in self.events}
        for sc in self.scenes:
            mp[sc.start_str] = sc.start_sec
        return mp

    def get_valid_end_map(self) -> Dict[str, float]:
        """Returns mapping of exact end timestamp strings to seconds."""
        mp = {ev.end_str: ev.end_sec for ev in self.events}
        for sc in self.scenes:
            mp[sc.end_str] = sc.end_sec
        return mp

    def format_for_llm(self) -> str:
        """Formats the structured screenplay with explicit Event IDs for accurate LLM grounding."""
        lines = [
            f"EPISODE NUMBER: {self.episode_number} (ID: {self.production_id})",
            f"TOTAL DURATION: {self.total_duration_str} ({self.total_duration_sec:.2f}s)",
            f"LOGLINE: {self.logline}",
            f"CONTINUITY NOTES: {self.continuity_notes}",
            "--- TIMECODED EVENTS ---",
        ]
        current_scene = None
        for ev in self.events:
            if ev.scene_number != current_scene:
                current_scene = ev.scene_number
                sc_obj = next((s for s in self.scenes if s.scene_number == current_scene), None)
                if sc_obj:
                    lines.append(
                        f"\n[SCENE {sc_obj.scene_number:02d}] {sc_obj.heading} "
                        f"({sc_obj.start_str} - {sc_obj.end_str})"
                    )
            tag = ev.event_type if ev.event_type != "DIALOGUE" else f"DIALOGUE:{ev.raw_header}"
            lines.append(
                f"[{ev.event_id}] ({ev.start_str} -> {ev.end_str}) [{tag}] {ev.text}"
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PromoSegment:
    """A single continuous time-slice from a specific episode."""
    episode_number: int
    start_timestamp: str
    end_timestamp: str
    start_sec: float = 0.0
    end_sec: float = 0.0
    duration_sec: float = 0.0
    start_event_id: Optional[str] = None
    end_event_id: Optional[str] = None
    segment_summary: str = ""
    validation_note: str = "UNVERIFIED"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PromoScores:
    """Individual 0-10 evaluation scores for a promo candidate."""
    hook_strength: float
    cliffhanger_strength: float
    standalone_context: float
    emotional_impact: float
    narrative_coherence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PromoCandidate:
    """Complete representation of a discovered, validated, and ranked promo."""
    promo_id: str
    rank: int
    episode_numbers: List[int]
    promo_type: str               # "SINGLE_SEGMENT", "MULTI_SEGMENT_SAME_EP", "CROSS_EPISODE"
    segments: List[PromoSegment]
    total_duration_sec: float
    hook_description: str
    cliffhanger_description: str
    standalone_context: str
    why_it_works: str
    scores: PromoScores
    final_score: float            # Weighted 0-100 score computed in Python
    confidence_level: str         # "HIGH", "MEDIUM", "LOW"
    validation_status: str        # "VALID", "VALID_SNAPPED", "NEEDS_REVIEW", "INVALID"
    review_notes: List[str] = field(default_factory=list)
    quality_flags: List[str] = field(default_factory=list)
    human_rating: Optional[str] = None  # "Good", "Average", "Bad" or None
    script_excerpt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
