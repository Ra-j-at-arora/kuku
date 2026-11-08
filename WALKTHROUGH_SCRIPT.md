# PromoAI — 3-Minute Video Walkthrough Script

**Target Duration:** ~2 minutes 45 seconds  
**Format:** Screen recording (Loom / OBS / Zoom) showing the PromoAI Dashboard, code structure, and 50-episode outputs in your own voice.

---

## 0:00 – 0:30 | Problem Framing & Why Problem 1
"Hi team! I'm [Your Name], and for the Kuku AI Generalist Assignment I selected **Problem 1: Find the Promos (Ads)** and built **PromoAI**.
At Kuku, editors manually watch every 1-to-2 minute microdrama episode to find 20-to-60 second promotional clips for Instagram Reels, TikTok, and YouTube Shorts.
When I inspected all 50 episodes of *I Married a Broken Soldier. He was a Duke*, I noticed two critical things:
First, these aren't standard screenplays—they are multi-track timecoded transcripts with 3,727 `CAMERA`, `ACTION`, `SOUND FX`, and `DIALOGUE` blocks that frequently overlap in time, plus formatting drift like Episode 6 using all-caps scene headers.
Second, LLMs are great at understanding dramatic betrayal and cliffhangers, but unreliable at timecode arithmetic. So I designed a strict neuro-symbolic pipeline separating AI narrative reasoning from deterministic Python verification."

## 0:30 – 1:15 | Architecture & How PromoAI Works (Show Code / Architecture)
"Let me quickly show how the pipeline works across four stages:
1. **Deterministic Ingestion (`src/parser.py`)**: Unpacks any screenplay ZIP, handles formatting anomalies case-insensitively, and indexes every valid block start and end timestamp into an $O(1)$ lookup table while tagging each block with a unique Event ID like `[E008]`.
2. **Two-Stage Discovery (`src/gemini_client.py` & `src/discovery.py`)**: Processes episodes individually via the swappable Gemini API using a strict JSON Schema (`EPISODE_DISCOVERY_SCHEMA`) and SHA-256 disk caching. It discovers single-segment clips, 2-segment intra-episode jump cuts that remove mid-scene filler, and cross-episode trailers—without ever stuffing all 50 episodes into one giant prompt.
3. **Python Verification Firewall (`src/validator.py`)**: Python verifies every AI-selected timestamp against the ground-truth screenplay. If an LLM drifts by half a second, Python snaps it to the nearest actual block boundary within 1.5s (`VALID_SNAPPED`) or recovers via the Event ID, and computes exact wall-clock duration between 20 and 60 seconds.
4. **Configurable Scoring & tIoU Deduplication (`src/scorer.py`)**: Python calculates the weighted score—30% Hook, 30% Cliffhanger, 20% Standalone Context, 10% Emotion, and 10% Coherence—and removes overlapping clips using Temporal Intersection-over-Union."

## 1:15 – 2:05 | Live Demo of the 50-Episode Results (Show Dashboard)
"Here is the PromoAI application loaded with our full 50-episode run:
- All **50 episodes** were processed with **0 failures**, producing **85 unique ranked promos** (`48 VALID` and `37 flagged as NEEDS_REVIEW`).
- Our **#1 Ranked Promo (`PROMO_001`, Score 9.64, 49.5s)** is a cross-episode trailer combining **Episode 1 (`00:33.70–00:54.60`)**—where Juliana in her bridal gown asks *'Why are you holding her hand?'* and her sister gloats *'Because he is marrying me'*—with **Episode 42 (`00:11.20–00:39.80`)**, where her scarred soldier husband strides into the cathedral in full ducal regalia and declares *'And yet, here I stand. That is the lie you built your life upon.'*
- If I click **Inspect** on any promo, I can see its 5 sub-scores, exact segment timestamps, validation notes, and the **verbatim timecoded screenplay excerpt** pulled directly from the source `.txt` file.
- I can also adjust the Stage 2 scoring sliders in real time or filter by episode, promo structure, or duration, and export `promos.json` and `promos.csv` in one click."

## 2:05 – 2:45 | Quality Checking: Where PromoAI Picks Badly & Scalability
"Finally, the brief asks: *Where does your tool pick badly?*
In the **Quality Audit** tab (`src/evaluator.py`), PromoAI automatically flags four failure modes:
- Take **Rank #84 (`PROMO_084`, Episode 6, Score 5.69)**: the window scanner picked `00:00.00–00:27.70`, which opens on a slow 3.5-second rainy walking shot in the woods and ends on Juliana asking *'That is it?'* at the cabin. Our evaluator caught both the slow opening and low-stakes ending, flagging it `WEAK_HOOK` and `NOT_SUITABLE_FOR_PROMOTION`.
- Or **Rank #85 (`PROMO_085`, Episode 17, Score 5.48)**: high emotional intensity fooled Stage 1, but the dialogue uses unresolved pronouns (*'You cannot promise that'* $\rightarrow$ *'Of them?'*) that fail the Cold Viewer Test (`MISSING_STANDALONE_CONTEXT`).
- Because episodes are processed independently with disk checkpointing, rate-limit backoff, and an environment-variable model selector (`GEMINI_MODEL`), Kuku can swap models or run PromoAI on a new 100-episode show pack in under two minutes. Thank you!"
