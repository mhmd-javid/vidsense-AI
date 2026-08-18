"""VideoAI — Streamlit UI.

Two sections:
  1. Process Video  — URL or upload, live pipeline status, clear errors.
  2. Chat With Video — question -> grounded answer + timestamp references +
     the transcript chunks that were used.

Run with:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make project root importable when launched via `streamlit run app/...`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from core.services import build_services  # noqa: E402
from core.utils import format_timestamp  # noqa: E402
from modules.ingestion.downloader import DownloadError  # noqa: E402
from modules.storage import db as dbmod  # noqa: E402
from modules.workflow.pipeline import STAGES  # noqa: E402

st.set_page_config(page_title="VideoAI — Chat with your video", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading models & services…")
def get_services():
    return build_services()


def rtl_markdown(text: str):
    """Render text with automatic direction (RTL for Persian, LTR otherwise)."""
    safe = (text or "").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    st.markdown(f"<div dir='auto' style='line-height:1.9'>{safe}</div>", unsafe_allow_html=True)


svc = get_services()

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🎬 VideoAI")
    st.caption("Local-first video intelligence · ASR → RAG → local LLM")

    ollama_ok = svc.llm.is_available()
    model_ok = ollama_ok and svc.llm.model_available()
    if model_ok:
        st.success(f"Ollama ready · {svc.llm.model}")
    elif ollama_ok:
        st.warning(f"Ollama up, but model '{svc.llm.model}' is missing.\n\n"
                   f"Run:  `ollama pull {svc.llm.model}`")
    else:
        st.error("Ollama not reachable. Start it (`ollama serve`) to enable chat.")

    with st.expander("Configuration", expanded=False):
        st.write(f"**ASR:** `{svc.cfg.asr.model_size}` / `{svc.cfg.asr.device}` / `{svc.cfg.asr.compute_type}`")
        st.write(f"**Embeddings:** `{svc.cfg.embedding.model_name}` @ `{svc.cfg.embedding.device}`")
        st.write(f"**LLM:** `{svc.cfg.llm.model}`")
        st.write(f"**Vector store:** `{svc.cfg.vectorstore.backend}`  ·  top_k=`{svc.cfg.rag.top_k}`")

    st.divider()
    st.subheader("Processed videos")
    videos = svc.db.list_videos()
    if not videos:
        st.caption("None yet — process a video to begin.")
    for v in videos:
        icon = "✅" if v["status"] == dbmod.STATUS_READY else ("⚠️" if v["status"] == dbmod.STATUS_ERROR else "⏳")
        st.write(f"{icon} {v['title'][:36]}  \n`{v['status']}` · {svc.db.count_chunks(v['video_id'])} chunks")


# --------------------------------------------------------------------------- #
# Main tabs
# --------------------------------------------------------------------------- #
tab_process, tab_chat = st.tabs(["📥  Process Video", "💬  Chat With Video"])

# ----- Tab 1: Process ------------------------------------------------------- #
with tab_process:
    st.header("Process a video")
    st.caption("Provide a URL (YouTube / Aparat / others) **or** upload a file. "
               "If a download fails, just upload the file manually.")

    col_url, col_up = st.columns(2)
    with col_url:
        url = st.text_input("Video URL", placeholder="https://www.youtube.com/watch?v=…  or Aparat link")
    with col_up:
        upload = st.file_uploader("…or upload a video/audio file",
                                  type=["mp4", "mkv", "webm", "mov", "avi", "mp3", "wav", "m4a"])

    start = st.button("▶  Start processing", type="primary", use_container_width=True)

    # Render the target pipeline as a reference checklist.
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
                    log_box.markdown("\n".join(logs[-10:]))

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
                svc.rag.invalidate(result.video_id)
                st.session_state["active_video"] = result.video_id
                st.success(
                    f"**{result.title}** is ready! "
                    f"Language: `{result.language}` · {result.num_chunks} chunks · "
                    f"{format_timestamp(result.duration)} · ASR on `{result.asr_device}`."
                )
                st.info("Switch to the **💬 Chat With Video** tab to ask questions.")
            elif result and not result.success:
                st.error(f"Processing failed: {result.error}")


# ----- Tab 2: Chat ---------------------------------------------------------- #
with tab_chat:
    st.header("Chat with your video")
    ready = svc.db.list_videos(ready_only=True)
    if not ready:
        st.info("No processed videos yet. Process one in the **📥 Process Video** tab.")
    else:
        labels = {f"{v['title'][:60]}  ·  {v['language'] or '?'}  ·  {v['video_id']}": v["video_id"]
                  for v in ready}
        default_idx = 0
        active = st.session_state.get("active_video")
        keys = list(labels.keys())
        if active:
            for i, k in enumerate(keys):
                if labels[k] == active:
                    default_idx = i
                    break
        chosen_label = st.selectbox("Video", keys, index=default_idx)
        video_id = labels[chosen_label]

        # Per-video chat history.
        hist_key = f"chat::{video_id}"
        history = st.session_state.setdefault(hist_key, [])

        for turn in history:
            with st.chat_message(turn["role"]):
                rtl_markdown(turn["content"])
                if turn.get("citations"):
                    with st.expander(f"📍 References ({len(turn['citations'])})"):
                        for c in turn["citations"]:
                            st.markdown(f"**({c['label']})** · score `{c['score']:.2f}`")
                            rtl_markdown(c["text"])
                            st.divider()

        question = st.chat_input("Ask something about this video…")
        if question:
            history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                rtl_markdown(question)
            with st.chat_message("assistant"):
                if not svc.llm.is_available():
                    st.error("Ollama is not reachable — start it to get answers.")
                else:
                    with st.spinner("Searching the video & thinking…"):
                        ans = svc.rag.answer(question, video_id)
                    rtl_markdown(ans.answer)
                    cites = [c.as_dict() for c in ans.citations]
                    if cites:
                        with st.expander(f"📍 References ({len(cites)})", expanded=True):
                            for c in cites:
                                st.markdown(f"**({c['label']})** · score `{c['score']:.2f}`")
                                rtl_markdown(c["text"])
                                st.divider()
                    history.append({"role": "assistant", "content": ans.answer, "citations": cites})
