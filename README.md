# PromoAI — AI-Powered Microdrama Promo Finder

**Kuku AI Generalist Assignment — Problem 1: Find the Promos (Ads)**

PromoAI is a modular, reusable two-stage AI + deterministic Python application that reads timecoded microdrama screenplays, discovers high-converting 20–60 second promotional clips (single continuous segments, multi-segment intra-episode cuts, and cross-episode trailers), verifies every timestamp against ground-truth screenplay boundaries, scores and ranks candidates using configurable weights, and flags failure modes for human editorial review.

---

## 1. Architecture Overview (Neuro-Symbolic Separation)

| Layer | Responsibility | Engine |
| :--- | :--- | :--- |
| **Stage 0: Screenplay Parser (`src/parser.py`)** | Unpacks `.zip` archives, normalizes case-sensitivity anomalies (e.g., Episode 6 ALL-CAPS headers), parses `CAMERA`, `ACTION`, `SOUND FX`, and `DIALOGUE` blocks into `EpisodeDocument` objects, and builds an $O(1)$ lookup table of valid start/end timestamps. | **Deterministic Python** |
| **Stage 1A & 1B: Candidate Discovery (`src/gemini_client.py`, `src/discovery.py`)** | Evaluates screenplay events for 3-second scroll-stop hooks, unresolved cliffhangers, standalone context (the "Cold Viewer Test"), and emotional stakes. Enforces structured JSON output (`EPISODE_DISCOVERY_SCHEMA`) with SHA-256 disk caching (`outputs/raw_model_responses/`). | **Gemini API (`google-genai` SDK) + Grounded Fallback** |
| **Stage 2A: Timestamp & Duration Validator (`src/validator.py`)** | Never blindly trusts AI timestamps. Cross-checks start/end timestamps and Event IDs (`[E001]`) against source screenplay blocks, snaps minor drift ($\le 1.5\text{s}$) to exact shot boundaries (`VALID_SNAPPED`), computes multi-track wall-clock durations ($20.0\text{s} \le D \le 60.0\text{s}$), and extracts verbatim script excerpts. | **Deterministic Python** |
| **Stage 2B: Scorer & Deduplicator (`src/scorer.py`)** | Computes configurable weighted scores ($0.30 H + 0.30 C + 0.20 S + 0.10 E + 0.10 N$), applies validation penalties, removes overlapping candidates via Temporal Intersection-over-Union ($\text{tIoU} \ge 0.60$), and assigns global ranks (`#1`–`#85`). | **Deterministic Python** |
| **Stage 3: Quality Audit & Streamlit UI (`src/evaluator.py`, `app.py`)** | Audits candidates for weak hooks, weak cliffhangers, and missing standalone context ("Where does the tool pick badly?"), and provides a Streamlit dashboard with human-in-the-loop review (`Good` / `Average` / `Bad`). | **Python + Streamlit** |

---

## 2. Project Structure

```text
promoai/
├── app.py                        # Streamlit web application (Dashboard, Upload, Filters, Script Inspector, Review)
├── run_pipeline.py               # CLI batch pipeline runner & exporter
├── requirements.txt              # Python dependencies
├── .env.example                  # Configurable environment variables (GEMINI_API_KEY, GEMINI_MODEL, weights)
├── .gitignore                    # Excludes .env and private Kuku screenplay files
├── WALKTHROUGH_SCRIPT.md         # 3-minute screen-recording script for submission
├── src/
│   ├── __init__.py
│   ├── config.py                 # Environment configuration & normalized scoring weights
│   ├── models.py                 # Typed dataclasses for Episodes, Events, Segments, and Promos
│   ├── parser.py                 # Case-insensitive screenplay parser & timestamp utilities
│   ├── gemini_client.py          # Google Gen AI SDK client, structured JSON schema, retries & caching
│   ├── discovery.py              # Stage 1A (intra-episode) & Stage 1B (cross-episode) discovery
│   ├── validator.py              # Deterministic timestamp verification, snapping & excerpt extraction
│   ├── scorer.py                 # Weighted scoring formula, tIoU deduplication & ranking
│   └── evaluator.py              # Automated failure-mode quality auditor & human review persistence
├── tests/
│   └── test_promoai.py           # 8 automated unit tests (parsing, durations, multi-seg, scoring, tIoU)
└── outputs/
    ├── promos.json               # All 85 ranked promo candidates from the 50-episode run
    ├── promos.csv                # Flat tabular export for editors and marketing teams
    ├── episode_analysis.json     # Per-episode structural and dramatic summary across all 50 episodes
    ├── evaluation_report.json    # Quality flag breakdown & documented poor selections
    └── raw_model_responses/      # 51 raw per-episode and cross-episode JSON outputs
```

---

## 3. Quickstart & Execution Instructions

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables (Optional if Using Pre-Computed Cache)
```bash
cp .env.example .env
# Edit .env to set GEMINI_API_KEY=<your_google_ai_studio_key> and GEMINI_MODEL=gemini-2.5-flash
```
*Note: You can also paste your `GEMINI_API_KEY` directly in the Streamlit sidebar at runtime, or explore the pre-computed 50-episode results immediately without an API key.*

### Step 3: Launch the Streamlit Application
```bash
streamlit run app.py
```

### Step 4: Run the CLI Batch Pipeline on Any Screenplay Pack
```bash
python3 run_pipeline.py --zip "path/to/screenplays.zip" --outputs outputs
```

### Step 5: Run the Automated Test Suite
```bash
python3 tests/test_promoai.py
```
