"""Configuration management for PromoAI using environment variables with sensible defaults."""
import os
from dataclasses import dataclass, field
from typing import Dict


def _load_dotenv_if_present(env_path: str = ".env") -> None:
    """Lightweight .env loader so python-dotenv is optional at runtime."""
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


_load_dotenv_if_present()


@dataclass
class ScoringWeights:
    """Configurable weights for Stage 2 Candidate Evaluation and Ranking."""
    hook_strength: float = float(os.getenv("WEIGHT_HOOK", "0.30"))
    cliffhanger_strength: float = float(os.getenv("WEIGHT_CLIFFHANGER", "0.30"))
    standalone_context: float = float(os.getenv("WEIGHT_CONTEXT", "0.20"))
    emotional_impact: float = float(os.getenv("WEIGHT_EMOTION", "0.10"))
    narrative_coherence: float = float(os.getenv("WEIGHT_COHERENCE", "0.10"))

    def normalized(self) -> Dict[str, float]:
        """Returns weights normalized to sum to 1.0."""
        total = (
            self.hook_strength
            + self.cliffhanger_strength
            + self.standalone_context
            + self.emotional_impact
            + self.narrative_coherence
        )
        if total <= 0:
            return {
                "hook_strength": 0.30,
                "cliffhanger_strength": 0.30,
                "standalone_context": 0.20,
                "emotional_impact": 0.10,
                "narrative_coherence": 0.10,
            }
        return {
            "hook_strength": self.hook_strength / total,
            "cliffhanger_strength": self.cliffhanger_strength / total,
            "standalone_context": self.standalone_context / total,
            "emotional_impact": self.emotional_impact / total,
            "narrative_coherence": self.narrative_coherence / total,
        }


@dataclass
class AppConfig:
    """Central configuration for PromoAI."""
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    min_promo_duration_sec: float = float(os.getenv("MIN_PROMO_DURATION_SEC", "20.0"))
    max_promo_duration_sec: float = float(os.getenv("MAX_PROMO_DURATION_SEC", "60.0"))
    snap_tolerance_sec: float = float(os.getenv("SNAP_TOLERANCE_SEC", "1.5"))
    dedup_iou_threshold: float = float(os.getenv("DEDUP_IOU_THRESHOLD", "0.60"))
    request_delay_sec: float = float(os.getenv("REQUEST_DELAY_SEC", "1.0"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    outputs_dir: str = field(default_factory=lambda: os.getenv("OUTPUTS_DIR", "outputs"))
    weights: ScoringWeights = field(default_factory=ScoringWeights)
