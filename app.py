"""
PromoAI - AI-Powered Microdrama Promo Finder
Streamlit Frontend Application (Part 6 of Kuku AI Generalist Assignment)

Features:
1. Dashboard with processing statistics
2. Upload screenplay ZIP (or load pre-processed outputs)
3. Analyze Episodes button with configurable Gemini model & scoring weights
4. Episode-wise progress bar & live status
5. Ranked promo results table
6. Individual promo details & verbatim screenplay excerpt inspector
7. Filters for episode, score, duration, promo type, and validation status
8. Episode analysis breakdown across all 50 episodes
9. JSON and CSV download buttons
10. Human review ('Good', 'Average', 'Bad') and Quality Evaluation failure-mode audit
"""
import io
import json
import os
import pandas as pd
import streamlit as st

from src.config import AppConfig, ScoringWeights
from src.parser import load_episodes_from_zip
from src.evaluator import save_human_rating
from run_pipeline import execute_full_pipeline


st.set_page_config(
    page_title="PromoAI — Microdrama Promo Finder",
    page_icon="🎬",
    layout="wide",
)


def _load_saved_outputs(outputs_dir: str):
    """Loads existing outputs/promos.json, episode_analysis.json, and evaluation_report.json if present."""
    promos_p = os.path.join(outputs_dir, "promos.json")
    ep_p = os.path.join(outputs_dir, "episode_analysis.json")
    eval_p = os.path.join(outputs_dir, "evaluation_report.json")
    csv_p = os.path.join(outputs_dir, "promos.csv")

    if not (os.path.exists(promos_p) and os.path.exists(ep_p)):
        return None

    with open(promos_p, "r", encoding="utf-8") as f:
        promos_data = json.load(f)
    with open(ep_p, "r", encoding="utf-8") as f:
        ep_data = json.load(f)
    eval_data = {}
    if os.path.exists(eval_p):
        with open(eval_p, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
    csv_bytes = b""
    if os.path.exists(csv_p):
        with open(csv_p, "rb") as f:
            csv_bytes = f.read()

    return {
        "stats": promos_data.get("processing_stats", {}),
        "promos": promos_data.get("promos", []),
        "duplicates": promos_data.get("duplicates_archived", []),
        "episodes": ep_data.get("episodes", []),
        "evaluation": eval_data,
        "promos_json_bytes": json.dumps(promos_data, indent=2, ensure_ascii=False).encode("utf-8"),
        "promos_csv_bytes": csv_bytes,
    }


def main():
    st.title("🎬 PromoAI — AI-Powered Microdrama Promo Finder")
    st.caption(
        "Automated Two-Stage Promo Discovery, Deterministic Timestamp Verification & Ranking Engine for Kuku Microdramas"
    )

    # Sidebar: Configuration, Upload, & Weights
    with st.sidebar:
        st.header("⚙️ Pipeline Configuration")
        api_key_input = st.text_input(
            "Gemini API Key (optional if cached)",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password",
            help="Set GEMINI_API_KEY in .env or paste here. Uses cached/deterministic analyzer if blank.",
        )
        model_name = st.selectbox(
            "Gemini Model",
            options=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
            index=0,
        )
        use_cache = st.checkbox("Use Cached Episode Responses", value=True)

        st.subheader("⚖️ Stage 2 Scoring Weights")
        w_hook = st.slider("Hook Strength (First 3s)", 0.0, 1.0, 0.30, 0.05)
        w_cliff = st.slider("Cliffhanger Strength", 0.0, 1.0, 0.30, 0.05)
        w_ctx = st.slider("Standalone Context", 0.0, 1.0, 0.20, 0.05)
        w_emo = st.slider("Emotional Impact", 0.0, 1.0, 0.10, 0.05)
        w_coh = st.slider("Narrative Coherence", 0.0, 1.0, 0.10, 0.05)

        st.subheader("⏱️ Duration Bounds (Seconds)")
        col_d1, col_d2 = st.columns(2)
        min_dur = col_d1.number_input("Min (s)", value=20.0, min_value=5.0, max_value=45.0, step=1.0)
        max_dur = col_d2.number_input("Max (s)", value=60.0, min_value=25.0, max_value=120.0, step=1.0)

        st.divider()
        st.subheader("📦 Upload Screenplay Pack")
        uploaded_zip = st.file_uploader("Upload Screenplay ZIP (.zip)", type=["zip"])
        analyze_btn = st.button("🚀 Analyze Episodes", type="primary", use_container_width=True)

    outputs_dir = os.getenv("OUTPUTS_DIR", "outputs")

    if analyze_btn:
        if uploaded_zip is None:
            st.error("Please upload a screenplay ZIP file before clicking 'Analyze Episodes'.")
        else:
            cfg = AppConfig(
                gemini_api_key=api_key_input,
                gemini_model=model_name,
                min_promo_duration_sec=min_dur,
                max_promo_duration_sec=max_dur,
                outputs_dir=outputs_dir,
                weights=ScoringWeights(
                    hook_strength=w_hook,
                    cliffhanger_strength=w_cliff,
                    standalone_context=w_ctx,
                    emotional_impact=w_emo,
                    narrative_coherence=w_coh,
                ),
            )
            with st.spinner("Unpacking and parsing screenplay files..."):
                episodes = load_episodes_from_zip(io.BytesIO(uploaded_zip.read()))

            prog_bar = st.progress(0.0, text=f"Initializing discovery across {len(episodes)} episodes...")
            status_box = st.empty()

            def _on_progress(idx, total, ep, analysis):
                prog_bar.progress(
                    idx / total,
                    text=f"Processed Episode {ep.episode_number}/{total} ({ep.total_duration_str}) — {len(analysis.get('candidates', []))} candidate(s)",
                )
                status_box.caption(f"Latest: Ep {ep.episode_number} — {ep.logline[:100]}...")

            execute_full_pipeline(
                episodes=episodes,
                config=cfg,
                use_cache=use_cache,
                progress_callback=_on_progress,
            )
            prog_bar.empty()
            status_box.success(f"Successfully processed all {len(episodes)} episodes!")

    state = _load_saved_outputs(outputs_dir)
    if state is None:
        st.info(
            "Upload a screenplay `.zip` pack in the sidebar and click **Analyze Episodes**, "
            "or run `python3 run_pipeline.py --zip <path.zip>` to populate `outputs/`."
        )
        return

    stats = state["stats"]
    promos = state["promos"]
    episodes_list = state["episodes"]
    eval_report = state["evaluation"]

    # 1. Top Dashboard Metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Episodes Processed", f"{stats.get('successfully_processed_episodes', 0)}/{stats.get('total_episodes_input', 0)}")
    c2.metric("Ranked Promos", stats.get("unique_ranked_promos", len(promos)))
    c3.metric("Valid (20–60s)", stats.get("valid_promos", 0))
    c4.metric("Needs Review", stats.get("promos_requiring_review", 0))
    c5.metric("Multi-Segment / Cross-Ep", stats.get("multi_segment_same_ep_promos", 0) + stats.get("cross_episode_promos", 0))
    c6.metric("Duplicates Filtered", stats.get("duplicates_removed", 0))

    # Download Bar
    dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 2])
    dl_col1.download_button(
        "⬇️ Download promos.json",
        data=state["promos_json_bytes"],
        file_name="promos.json",
        mime="application/json",
        use_container_width=True,
    )
    dl_col2.download_button(
        "⬇️ Download promos.csv",
        data=state["promos_csv_bytes"],
        file_name="promos.csv",
        mime="text/csv",
        use_container_width=True,
    )
    dl_col3.caption(
        f"**Active Model:** `{stats.get('model_configured', 'gemini-2.5-flash')}` | "
        f"**Engine Mode:** `{stats.get('engine_mode', 'cached')}` | "
        f"**Run Time:** `{stats.get('processing_time_sec', 0)}s`"
    )

    tab_promos, tab_inspector, tab_episodes, tab_eval = st.tabs(
        [
            "🏆 Ranked Promos & Filters",
            "🔍 Promo Detail & Script Inspector",
            "📚 Episode-by-Episode Analysis (50 Eps)",
            "⚠️ Quality Evaluation & Human Review",
        ]
    )

    # TAB 1: Ranked Promos with Filters
    with tab_promos:
        st.subheader("Filter Ranked Promo Candidates")
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        all_eps = sorted({ep for p in promos for ep in p.get("episode_numbers", [])})
        selected_eps = f_col1.multiselect("Filter by Episode(s)", options=all_eps, default=[])
        min_score_filter = f_col2.slider("Minimum Final Score (0–10)", 0.0, 10.0, 0.0, 0.5)
        dur_range = f_col3.slider("Duration Range (sec)", 0.0, 70.0, (20.0, 60.0), 1.0)
        status_filter = f_col4.multiselect(
            "Validation Status",
            options=["VALID", "VALID_SNAPPED", "NEEDS_REVIEW", "INVALID"],
            default=["VALID", "VALID_SNAPPED", "NEEDS_REVIEW"],
        )

        filtered_promos = [
            p for p in promos
            if (not selected_eps or any(e in selected_eps for e in p.get("episode_numbers", [])))
            and p.get("final_score", 0.0) >= min_score_filter
            and dur_range[0] <= p.get("total_duration_sec", 0.0) <= dur_range[1]
            and (not status_filter or p.get("validation_status") in status_filter)
        ]

        st.write(f"Showing **{len(filtered_promos)}** of **{len(promos)}** ranked promos:")
        table_rows = []
        for p in filtered_promos:
            segs_txt = " | ".join(
                f"Ep {s['episode_number']}: {s['start_timestamp']}–{s['end_timestamp']} ({s['duration_sec']}s)"
                for s in p.get("segments", [])
            )
            sc = p.get("scores", {})
            table_rows.append(
                {
                    "Rank": p.get("rank"),
                    "Promo ID": p.get("promo_id"),
                    "Type": p.get("promo_type"),
                    "Episode(s)": ", ".join(str(x) for x in p.get("episode_numbers", [])),
                    "Segments & Timestamps": segs_txt,
                    "Duration (s)": p.get("total_duration_sec"),
                    "Final Score": p.get("final_score"),
                    "Hook (30%)": sc.get("hook_strength"),
                    "Cliffhanger (30%)": sc.get("cliffhanger_strength"),
                    "Context (20%)": sc.get("standalone_context"),
                    "Emotion (10%)": sc.get("emotional_impact"),
                    "Coherence (10%)": sc.get("narrative_coherence"),
                    "Status": p.get("validation_status"),
                    "Human Rating": p.get("human_rating") or "Unrated",
                }
            )
        if table_rows:
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        else:
            st.warning("No promos match the current filter criteria.")

    # TAB 2: Individual Promo Details & Verbatim Script Excerpt
    with tab_inspector:
        promo_options = {
            f"#{p['rank']} [{p['promo_id']}] Ep {','.join(map(str, p['episode_numbers']))} — Score {p['final_score']} ({p['total_duration_sec']}s)": p
            for p in promos
        }
        chosen_label = st.selectbox("Select Promo Candidate to Inspect", options=list(promo_options.keys()))
        if chosen_label:
            p = promo_options[chosen_label]
            sc = p.get("scores", {})
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Final Score", f"{p['final_score']}/10")
            m2.metric("Hook (30%)", f"{sc.get('hook_strength')}/10")
            m3.metric("Cliffhanger (30%)", f"{sc.get('cliffhanger_strength')}/10")
            m4.metric("Context (20%)", f"{sc.get('standalone_context')}/10")
            m5.metric("Duration", f"{p['total_duration_sec']}s")
            m6.metric("Validation", p["validation_status"])

            st.markdown("#### 🎯 Narrative Breakdown")
            st.markdown(f"- **Opening 3s Hook:** {p.get('hook_description')}")
            st.markdown(f"- **Ending Cliffhanger:** {p.get('cliffhanger_description')}")
            st.markdown(f"- **Standalone Context (Cold Viewer Test):** {p.get('standalone_context')}")
            st.markdown(f"- **Why This Promo Works:** {p.get('why_it_works')}")

            st.markdown("#### 🎞️ Exact Segment Timestamps")
            st.table(pd.DataFrame(p.get("segments", [])))

            if p.get("quality_flags"):
                st.warning(f"**Quality Flags:** {', '.join(p['quality_flags'])}")
            st.info("**Validation & Review Notes:**\n- " + "\n- ".join(p.get("review_notes", [])))

            with st.expander("📜 View Verbatim Screenplay Excerpt (Verified Against Source Text)", expanded=True):
                st.code(p.get("script_excerpt", "No excerpt available."), language="text")

    # TAB 3: Episode Analysis
    with tab_episodes:
        st.subheader("50-Episode Processing & Dramatic Analysis Summary")
        ep_df = pd.DataFrame(
            [
                {
                    "Episode": e.get("episode_number"),
                    "Production ID": e.get("production_id"),
                    "Duration": e.get("total_duration_str"),
                    "Scenes": e.get("scene_count"),
                    "Timed Events": e.get("event_count"),
                    "Strong Material?": "✅ Yes" if e.get("has_strong_promo_material") else "⚠️ Transitional",
                    "Candidates Found": e.get("candidates_discovered"),
                    "Dominant Emotions": ", ".join(e.get("dominant_emotions", [])),
                    "Episode Logline": e.get("logline"),
                }
                for e in episodes_list
            ]
        )
        st.dataframe(ep_df, use_container_width=True, hide_index=True)

    # TAB 4: Quality Evaluation & Human-in-the-Loop Review
    with tab_eval:
        st.subheader("Part 7: Where Does the Tool Pick Badly? (Failure Mode Audit)")
        st.markdown(
            "PromoAI automatically audits candidates for **weak 3s hooks**, **weak cliffhangers**, "
            "**missing standalone context (mid-season lore traps)**, and **timestamp/duration anomalies**."
        )
        col_ev1, col_ev2 = st.columns(2)
        with col_ev1:
            st.markdown("##### Automated Quality Flag Breakdown")
            flags_dict = eval_report.get("quality_flag_breakdown", {})
            if flags_dict:
                st.bar_chart(pd.Series(flags_dict, name="Count"))
            else:
                st.write("No quality flags triggered.")

        with col_ev2:
            st.markdown("##### Documented Weak / Lower-Ranked Selections (Where AI Picks Badly)")
            poor_picks = eval_report.get("documented_poor_selections", [])
            if poor_picks:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Promo ID": x["promo_id"],
                                "Rank": x["rank"],
                                "Episode(s)": ", ".join(map(str, x["episode_numbers"])),
                                "Score": x["final_score"],
                                "Flags": ", ".join(x["quality_flags"]),
                                "Failure / Review Explanation": x["why_it_failed_or_needs_review"],
                            }
                            for x in poor_picks
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        st.divider()
        st.subheader("👩‍⚖️ Human Reviewer Classification (Good / Average / Bad)")
        r_col1, r_col2, r_col3, r_col4 = st.columns([2, 1, 2, 1])
        target_promo_id = r_col1.selectbox(
            "Select Promo ID",
            options=[p["promo_id"] for p in promos],
        )
        rating_choice = r_col2.selectbox("Human Rating", options=["Good", "Average", "Bad"])
        reviewer_comment = r_col3.text_input("Reviewer Note (optional)")
        if r_col4.button("💾 Save Rating", use_container_width=True):
            ok = save_human_rating(outputs_dir, target_promo_id, rating_choice, reviewer_comment)
            if ok:
                st.success(f"Saved human rating '{rating_choice}' for {target_promo_id}!")
                st.rerun()


if __name__ == "__main__":
    main()
