"""VidSense — Streamlit UI for the Persian transcription pipeline.

Two sections:
  1. Process Video — URL or upload, live staged pipeline progress.
  2. Transcript    — RTL segment view with speaker chips, time ranges,
     low-confidence highlighting, and JSON / SRT / VTT downloads.

Run with:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make project root importable when launched via `streamlit run app/...`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from core.services import build_services  # noqa: E402
from modules.ingestion.downloader import DownloadError  # noqa: E402
from modules.storage import db as dbmod  # noqa: E402
from modules.workflow.pipeline import STAGES  # noqa: E402

st.set_page_config(page_title="VidSense — Persian transcription", page_icon="🎧", layout="wide")


@st.cache_resource(show_spinner="Loading services…")
def get_services():
    return build_services()


def _mmss(seconds) -> str:
    try:
        s = int(float(seconds))
    except (TypeError, ValueError):
        return "00:00"
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def rtl_markdown(text: str, low: bool = False):
    """Render text with automatic direction (RTL for Persian)."""
    safe = (text or "").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    bg = "background:#4a3a00;padding:4px 8px;border-radius:6px;" if low else ""
    st.markdown(
        f"<div dir='auto' style='line-height:1.9;{bg}'>{safe}</div>",
        unsafe_allow_html=True,
    )


svc = get_services()

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🎧 VidSense")
    st.caption("Local-first Persian speech-to-text · VAD → ASR → align → diarize")

    with st.expander("Configuration", expanded=False):
        st.write(f"**ASR:** `{svc.cfg.asr.model_size}` / `{svc.cfg.asr.device}` / `{svc.cfg.asr.compute_type}`")
        st.write(f"**Language:** `{svc.cfg.asr.language or 'auto'}`")
        st.write(f"**VAD:** `{svc.cfg.vad.method}` · chunk `{svc.cfg.vad.chunk_size}s`")
        st.write(f"**Alignment:** `{'on' if svc.cfg.alignment.enabled else 'off'}`")
        st.write(f"**Diarization:** `{'on' if svc.cfg.diarization.enabled else 'off'}`")

    st.divider()
    st.subheader("Processed videos")
    videos = svc.db.list_videos()
    if not videos:
        st.caption("None yet — process a video to begin.")
    for v in videos:
        icon = "✅" if v["status"] == dbmod.STATUS_READY else ("⚠️" if v["status"] == dbmod.STATUS_ERROR else "⏳")
        q = v.get("quality_score")
        meta = f"`{v['status']}`"
        if q is not None:
            meta += f" · q={q:.2f}"
        st.write(f"{icon} {(v['title'] or v['video_id'])[:36]}  \n{meta}")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_process, tab_transcript = st.tabs(["📥  Process Video", "📄  Transcript"])

# ----- Tab 1: Process ------------------------------------------------------- #
with tab_process:
    st.header("Process a video")
    st.caption("Provide a URL (YouTube / Aparat / direct link) **or** upload a file. "
               "If a download fails, just upload the file manually.")

    col_url, col_up = st.columns(2)
    with col_url:
        url = st.text_input("Video URL", placeholder="https://www.youtube.com/watch?v=…  or Aparat link")
    with col_up:
        upload = st.file_uploader("…or upload a video/audio file",
                                  type=["mp4", "mkv", "webm", "mov", "avi", "mp3", "wav", "m4a"])

    start = st.button("▶  Start transcription", type="primary", use_container_width=True)

    with st.expander("Pipeline stages", expanded=False):
        st.markdown("  →  ".join(f"**{label}**" for _, label in STAGES))

    if start:
        if not url and not upload:
            st.warning("Enter a URL or upload a file first.")
        else:
            prog = st.progress(0.0, text="Starting…")
            log_box = st.container()
            logs: list[str] = []
            stage_index = {k: i for i, (k, _) in enumerate(STAGES)}
            n = len(STAGES)

            def cb(stage: str, msg: str, frac):
                if stage in stage_index:
                    base = stage_index[stage] / n
                    overall = base + (1.0 / n) * (frac if frac is not None else 1.0)
                    prog.progress(min(1.0, overall), text=msg)
                logs.append(f"- {msg}")
                with log_box:
                    log_box.markdown("\n".join(logs[-12:]))

            try:
                if upload is not None:
                    result = svc.processing.process_upload(upload.getvalue(), upload.name, progress_cb=cb)
                else:
                    result = svc.processing.process_url(url, progress_cb=cb)
            except DownloadError as exc:
                result = None
                st.error(str(exc))

            if result and result.success:
                prog.progress(1.0, text="Ready ✅")
                st.session_state["active_video"] = result.video_id
                prob = result.language_probability
                prob_txt = f" (p={prob:.2f})" if isinstance(prob, (int, float)) else ""
                q = result.quality_score
                q_txt = f" · quality `{q:.2f}`" if isinstance(q, (int, float)) else ""
                st.success(
                    f"**{result.title}** transcribed! "
                    f"Language `{result.language}`{prob_txt} · {result.num_segments} segments · "
                    f"{result.num_speakers} speaker(s){q_txt} · ASR on `{result.asr_device}`."
                )
                st.info("Open the **📄 Transcript** tab to read and download it.")
            elif result and not result.success:
                st.error(f"Processing failed: {result.error}")


# ----- Tab 2: Transcript ---------------------------------------------------- #
with tab_transcript:
    st.header("Transcript")
    ready = svc.db.list_videos(ready_only=True)
    if not ready:
        st.info("No transcripts yet. Process a video in the **📥 Process Video** tab.")
    else:
        labels = {f"{(v['title'] or v['video_id'])[:60]}  ·  {v['language'] or '?'}  ·  {v['video_id']}": v
                  for v in ready}
        keys = list(labels.keys())
        active = st.session_state.get("active_video")
        default_idx = next((i for i, k in enumerate(keys) if labels[k]["video_id"] == active), 0)
        chosen = st.selectbox("Video", keys, index=default_idx)
        v = labels[chosen]

        tpath = v.get("transcript_path")
        data = None
        if tpath and Path(tpath).exists():
            try:
                data = json.loads(Path(tpath).read_text(encoding="utf-8"))
            except Exception as exc:
                st.error(f"Could not read transcript JSON: {exc}")
        if not data:
            st.warning("Transcript file not found on disk.")
        else:
            # Metadata row.
            c1, c2, c3, c4, c5 = st.columns(5)
            prob = data.get("language_probability")
            c1.metric("Language", f"{data.get('language', '?')}",
                      f"p={prob:.2f}" if isinstance(prob, (int, float)) else None)
            c2.metric("Duration", _mmss(data.get("duration") or 0))
            c3.metric("Segments", len(data.get("segments", [])))
            c4.metric("Speakers", data.get("num_speakers", 1))
            q = data.get("quality_score")
            c5.metric("Quality", f"{q:.2f}" if isinstance(q, (int, float)) else "—")

            # Downloads.
            d1, d2, d3 = st.columns(3)
            for col, key, label, mime in (
                (d1, "transcript_path", "⬇ JSON", "application/json"),
                (d2, "srt_path", "⬇ SRT", "text/plain"),
                (d3, "vtt_path", "⬇ VTT", "text/vtt"),
            ):
                p = v.get(key)
                if p and Path(p).exists():
                    col.download_button(
                        label, data=Path(p).read_bytes(), file_name=Path(p).name,
                        mime=mime, use_container_width=True,
                    )

            st.divider()
            multi = (data.get("num_speakers", 1) or 1) > 1
            for seg in data.get("segments", []):
                rng = f"{_mmss(seg['start'])} – {_mmss(seg['end'])}"
                conf = seg.get("confidence")
                low = seg.get("low_confidence", False)
                head = f"`{rng}`"
                if multi:
                    head += f"  ·  **{seg.get('speaker', 'SPEAKER_00')}**"
                if isinstance(conf, (int, float)):
                    head += f"  ·  conf `{conf:.2f}`" + ("  ⚠️" if low else "")
                st.markdown(head)
                rtl_markdown(seg.get("text", ""), low=low)
