"""Deterministic screenplay parser and timestamp utility functions for PromoAI."""
import io
import os
import re
import zipfile
from typing import List, Optional, Tuple, Dict
from src.models import ScriptEvent, SceneBlock, EpisodeDocument


NON_DIALOGUE_HEADERS = {
    "CAMERA": "CAMERA",
    "ACTION": "ACTION",
    "SOUND FX": "SOUND_FX",
    "SFX": "SOUND_FX",
    "MUSIC": "SOUND_FX",
    "VFX": "CAMERA",
}

HEADER_EPISODE_RE = re.compile(r"^EPISODE\s+NUMBER\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
HEADER_LOGLINE_RE = re.compile(r"^EPISODE\s+LOGLINE\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
HEADER_CONTINUITY_RE = re.compile(r"^CONTINUITY\s+NOTES\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)

SCENE_SPLIT_RE = re.compile(r"(?=^SCENE\s+\d+)", re.IGNORECASE | re.MULTILINE)
SCENE_NUM_RE = re.compile(r"^SCENE\s+(\d+)", re.IGNORECASE | re.MULTILINE)
SCENE_HEADING_RE = re.compile(r"^Scene\s+heading\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
SCENE_START_RE = re.compile(r"^Scene\s+start\s*:\s*([0-9:.]+)", re.IGNORECASE | re.MULTILINE)
SCENE_END_RE = re.compile(r"^Scene\s+end\s*:\s*([0-9:.]+)", re.IGNORECASE | re.MULTILINE)

EVENT_BLOCK_RE = re.compile(
    r"\*\*(?P<header>.+?)\*\*\s*\n"
    r"Start:\s*\(?(?P<start>[0-9:.]+)\)?\s*\|\s*End:\s*\(?(?P<end>[0-9:.]+)\)?\s*\n"
    r"(?P<text>.+?)(?=\n\s*\*\*|\n\s*SCENE\s+\d+|\n\s*---\s*SCRIPT\s+END|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def parse_timestamp_to_seconds(ts_str: str) -> float:
    """
    Parses a timestamp string ('MM:SS.xx' or 'HH:MM:SS.xx') into float seconds.
    Raises ValueError if the timestamp is malformed or negative.
    """
    if not ts_str or not isinstance(ts_str, str):
        raise ValueError(f"Empty or non-string timestamp: {ts_str!r}")
    cleaned = ts_str.strip().strip("()").strip()
    parts = cleaned.split(":")
    try:
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            if minutes < 0 or seconds < 0 or seconds >= 60.0:
                raise ValueError(f"Invalid minute/second values in {ts_str!r}")
            return round(minutes * 60.0 + seconds, 3)
        elif len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            if hours < 0 or minutes < 0 or minutes >= 60 or seconds < 0 or seconds >= 60.0:
                raise ValueError(f"Invalid hour/minute/second values in {ts_str!r}")
            return round(hours * 3600.0 + minutes * 60.0 + seconds, 3)
    except ValueError as exc:
        raise ValueError(f"Cannot parse timestamp {ts_str!r}: {exc}") from exc
    raise ValueError(f"Unsupported timestamp format: {ts_str!r}")


def format_seconds_to_timestamp(seconds: float) -> str:
    """Formats float seconds back to standard 'MM:SS.xx' string."""
    if seconds < 0:
        seconds = 0.0
    mins = int(seconds // 60)
    rem = seconds - (mins * 60)
    return f"{mins:02d}:{rem:05.2f}"


def _extract_episode_identifiers(raw_text: str, filename: str) -> Tuple[int, str]:
    """Extracts sequential episode number and production ID from header and filename."""
    ep_num: Optional[int] = None
    prod_id: str = "UNKNOWN"

    # Check filename patterns like 'episode-001-Episode 1671703.txt'
    fname_match = re.search(r"episode[-_\s]*(\d+)", filename, re.IGNORECASE)
    if fname_match:
        ep_num = int(fname_match.group(1))

    prod_match = re.search(r"Episode\s+(\d{4,})", filename, re.IGNORECASE)
    if prod_match:
        prod_id = prod_match.group(1)

    header_match = HEADER_EPISODE_RE.search(raw_text)
    if header_match:
        header_val = header_match.group(1).strip()
        num_in_header = re.search(r"(\d+)", header_val)
        if num_in_header and ep_num is None:
            ep_num = int(num_in_header.group(1))

    if ep_num is None:
        raise ValueError(f"Could not determine episode number for file {filename!r}")
    return ep_num, prod_id


def _classify_header(raw_header_str: str) -> Tuple[str, Optional[str], Optional[str], str]:
    """
    Classifies a block header like 'MATRIARCH (O.S.) (Firm):' or 'CAMERA:'.
    Returns (event_type, speaker, delivery_modifier, cleaned_raw_header).
    """
    cleaned = raw_header_str.strip().rstrip(":").strip()
    paren_matches = re.findall(r"\(([^)]+)\)", cleaned)
    base_name = re.sub(r"\s*\([^)]*\)", "", cleaned).strip()
    upper_base = base_name.upper()

    if upper_base in NON_DIALOGUE_HEADERS:
        return NON_DIALOGUE_HEADERS[upper_base], None, None, cleaned

    modifier = ", ".join(m.strip() for m in paren_matches) if paren_matches else None
    return "DIALOGUE", base_name, modifier, cleaned


def parse_screenplay_text(raw_text: str, filename: str = "episode-001.txt") -> EpisodeDocument:
    """
    Parses a single episode screenplay file into a structured EpisodeDocument.
    Preserves all original text, timestamps, and scene hierarchy without fabricating content.
    """
    warnings: List[str] = []
    ep_num, prod_id = _extract_episode_identifiers(raw_text, filename)

    logline_m = HEADER_LOGLINE_RE.search(raw_text)
    logline = logline_m.group(1).strip() if logline_m else ""
    if not logline:
        warnings.append("Missing EPISODE LOGLINE header.")

    cont_m = HEADER_CONTINUITY_RE.search(raw_text)
    continuity_notes = cont_m.group(1).strip() if cont_m else ""

    if "--- SCRIPT START ---" in raw_text:
        script_body = raw_text.split("--- SCRIPT START ---", 1)[1]
    else:
        warnings.append("Missing --- SCRIPT START --- delimiter; parsing full text.")
        script_body = raw_text

    if "--- SCRIPT END ---" in script_body:
        script_body = script_body.split("--- SCRIPT END ---", 1)[0]
    else:
        warnings.append("Missing --- SCRIPT END --- delimiter.")

    scene_chunks = [c.strip() for c in SCENE_SPLIT_RE.split(script_body) if c.strip()]
    scenes: List[SceneBlock] = []
    all_events: List[ScriptEvent] = []
    character_set: List[str] = []
    event_counter = 1

    if not scene_chunks:
        scene_chunks = [script_body]

    for idx, chunk in enumerate(scene_chunks, start=1):
        sc_num_m = SCENE_NUM_RE.search(chunk)
        sc_num = int(sc_num_m.group(1)) if sc_num_m else idx

        sc_head_m = SCENE_HEADING_RE.search(chunk)
        sc_heading = sc_head_m.group(1).strip() if sc_head_m else f"SCENE {sc_num:02d}"

        sc_start_m = SCENE_START_RE.search(chunk)
        sc_end_m = SCENE_END_RE.search(chunk)

        scene_events: List[ScriptEvent] = []
        for match in EVENT_BLOCK_RE.finditer(chunk):
            raw_hdr = match.group("header")
            s_str = match.group("start").strip()
            e_str = match.group("end").strip()
            content_raw = match.group("text").strip()

            # Strip trailing 'ACTIONS AND DIALOGUE:' or 'Actions and dialogue:' if attached
            content_lines = [
                ln.strip()
                for ln in content_raw.splitlines()
                if ln.strip() and not re.match(r"^Actions\s+and\s+dialogue\s*:", ln.strip(), re.IGNORECASE)
            ]
            text_clean = " ".join(content_lines)

            try:
                s_sec = parse_timestamp_to_seconds(s_str)
                e_sec = parse_timestamp_to_seconds(e_str)
            except ValueError as err:
                warnings.append(f"Event timestamp parse error in scene {sc_num}: {err}")
                continue

            if e_sec < s_sec:
                warnings.append(
                    f"Event E{event_counter:03d} has end ({e_str}) < start ({s_str}); swapping."
                )
                s_sec, e_sec = e_sec, s_sec
                s_str, e_str = e_str, s_str

            ev_type, speaker, modifier, cleaned_hdr = _classify_header(raw_hdr)
            if speaker and speaker not in character_set:
                character_set.append(speaker)

            ev = ScriptEvent(
                event_id=f"E{event_counter:03d}",
                scene_number=sc_num,
                event_type=ev_type,
                raw_header=cleaned_hdr,
                speaker=speaker,
                delivery_modifier=modifier,
                start_str=s_str,
                end_str=e_str,
                start_sec=s_sec,
                end_sec=e_sec,
                duration_sec=round(e_sec - s_sec, 3),
                text=text_clean,
            )
            scene_events.append(ev)
            all_events.append(ev)
            event_counter += 1

        if sc_start_m:
            sc_s_str = sc_start_m.group(1).strip()
            sc_s_sec = parse_timestamp_to_seconds(sc_s_str)
        elif scene_events:
            sc_s_str = scene_events[0].start_str
            sc_s_sec = scene_events[0].start_sec
        else:
            sc_s_str, sc_s_sec = "00:00.00", 0.0

        if sc_end_m:
            sc_e_str = sc_end_m.group(1).strip()
            sc_e_sec = parse_timestamp_to_seconds(sc_e_str)
        elif scene_events:
            sc_e_str = max(scene_events, key=lambda x: x.end_sec).end_str
            sc_e_sec = max(x.end_sec for x in scene_events)
        else:
            sc_e_str, sc_e_sec = sc_s_str, sc_s_sec

        scenes.append(
            SceneBlock(
                scene_number=sc_num,
                heading=sc_heading,
                start_str=sc_s_str,
                end_str=sc_e_str,
                start_sec=sc_s_sec,
                end_sec=sc_e_sec,
                events=scene_events,
            )
        )

    total_dur_sec = 0.0
    total_dur_str = "00:00.00"
    if all_events:
        max_ev = max(all_events, key=lambda x: x.end_sec)
        total_dur_sec = max_ev.end_sec
        total_dur_str = max_ev.end_str
    elif scenes:
        total_dur_sec = max(s.end_sec for s in scenes)
        total_dur_str = format_seconds_to_timestamp(total_dur_sec)

    return EpisodeDocument(
        episode_number=ep_num,
        production_id=prod_id,
        source_filename=os.path.basename(filename),
        logline=logline,
        continuity_notes=continuity_notes,
        total_duration_sec=total_dur_sec,
        total_duration_str=total_dur_str,
        scenes=scenes,
        events=all_events,
        characters=character_set,
        parse_warnings=warnings,
    )


def load_episodes_from_zip(zip_path_or_bytes) -> List[EpisodeDocument]:
    """Loads and parses all episode .txt files from a ZIP path or BytesIO object."""
    episodes: List[EpisodeDocument] = []
    with zipfile.ZipFile(zip_path_or_bytes, "r") as zf:
        txt_names = [
            name for name in zf.namelist()
            if name.lower().endswith(".txt") and not os.path.basename(name).startswith(".")
        ]
        for name in sorted(txt_names):
            raw_bytes = zf.read(name)
            raw_text = raw_bytes.decode("utf-8", errors="replace")
            doc = parse_screenplay_text(raw_text, filename=os.path.basename(name))
            episodes.append(doc)
    episodes.sort(key=lambda x: x.episode_number)
    return episodes


def load_episodes_from_directory(dir_path: str) -> List[EpisodeDocument]:
    """Loads and parses all episode .txt files from a directory tree."""
    episodes: List[EpisodeDocument] = []
    for root, _, files in os.walk(dir_path):
        for f in sorted(files):
            if f.lower().endswith(".txt") and not f.startswith("."):
                full_path = os.path.join(root, f)
                with open(full_path, "r", encoding="utf-8", errors="replace") as fp:
                    raw_text = fp.read()
                episodes.append(parse_screenplay_text(raw_text, filename=f))
    episodes.sort(key=lambda x: x.episode_number)
    return episodes


def extract_script_excerpt(episode: EpisodeDocument, start_sec: float, end_sec: float) -> str:
    """Extracts verbatim lines from an EpisodeDocument falling within [start_sec, end_sec]."""
    selected = [
        ev for ev in episode.events
        if ev.end_sec > start_sec + 0.05 and ev.start_sec < end_sec - 0.05
    ]
    lines = []
    for ev in selected:
        prefix = f"[{ev.start_str}-{ev.end_str}] {ev.raw_header}:"
        lines.append(f"{prefix} {ev.text}")
    return "\n".join(lines)
