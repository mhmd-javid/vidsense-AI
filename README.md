# 🎬 VideoAI — Local-First Video Intelligence MVP

Give the system a video → it extracts the knowledge inside it → you chat with the
video and get answers **grounded in the transcript, every answer citing the
timestamps** it came from.

Everything runs **locally**. No paid APIs, no cloud calls for ASR, embeddings, or
the LLM. Built and tested on Windows with an **NVIDIA T1000 (4 GB VRAM)**.

```
video (URL or file) ─► download ─► extract audio ─► ASR (Whisper) ─►
        chunk ─► embed ─► store (SQLite + vectors) ─► RAG chat with citations
```

---

## Table of contents
1. [What it does](#what-it-does)
2. [How the 4 GB VRAM budget is respected](#how-the-4-gb-vram-budget-is-respected)
3. [Prerequisites](#prerequisites)
4. [Setup](#setup)
5. [Running the app](#running-the-app)
6. [Command-line pipeline / tests](#command-line-pipeline--tests)
7. [Configuration](#configuration)
8. [Project layout](#project-layout)
9. [Models used & why](#models-used--why)
10. [Known limitations](#known-limitations)

---

## What it does

- **Ingest** a video from a URL (YouTube, and *best-effort* Aparat / any
  yt-dlp-supported site) **or** a local file upload.
- **Extract** 16 kHz mono audio with a bundled FFmpeg (no system install).
- **Transcribe** with faster-whisper, producing segments with **timestamps**,
  detected language, and language probability. Works for **Persian** and other
  languages.
- **Chunk** the many tiny ASR segments into ~30 s semantically-coherent windows,
  preserving accurate `start`/`end` times and preferring sentence boundaries.
- **Embed** each chunk with a **multilingual** model (Persian-capable).
- **Store** video metadata, transcript, chunks, and embeddings in **SQLite**
  (schema is PostgreSQL-migratable).
- **Chat** via a RAG pipeline that answers **only** from retrieved chunks and
  **always cites timestamps** (e.g. `[02:10–02:35]`). If the transcript doesn't
  contain the answer, it says so instead of hallucinating.

The pipeline is a **deterministic** sequence of stages (no agents, no LangGraph),
but every stage sits behind a clean interface so an agent layer could be added
later.

---

## How the 4 GB VRAM budget is respected

The three heavy models are **never GPU-resident at the same time**:

| Stage        | Runs on | VRAM        |
|--------------|---------|-------------|
| ASR (Whisper `small`, int8) | **CPU** | 0 MiB (faster than real time) |
| Embeddings (e5-small)       | **CPU** | 0 MiB |
| LLM (Qwen2.5-3B via Ollama) | **GPU** | ~2.1 GB, only during chat |

Measured on the T1000 (`tests/measure_vram.py`):

```
Baseline VRAM (no LLM):  888 / 4096 MiB   (desktop only)
VRAM with LLM loaded  : 2994 / 4096 MiB
==> LLM footprint ≈ 2106 MiB   (headroom ≈ 1102 MiB)
```

Because ASR and embeddings are on CPU, the only GPU consumer is the LLM, so there
is **zero contention** and comfortable headroom on 4 GB.

> **GPU ASR is optional.** faster-whisper *can* use the GPU, but that needs the
> CUDA cuBLAS + cuDNN runtime DLLs (~1 GB download). They're not required — the
> code auto-detects their absence and runs ASR on CPU, which is already faster
> than real time for the `small`/`int8` model. See
> [Configuration](#configuration) to opt in.

---

## Prerequisites

- **Python 3.11** (tested on 3.11.9), 64-bit.
- **Ollama** for the local LLM — <https://ollama.com/download>.
- **FFmpeg**: *not required* — a static binary ships via `imageio-ffmpeg`.
- **GPU (optional)**: any NVIDIA card works for the LLM through Ollama. VideoAI
  runs fine on CPU-only machines too (the LLM will just be slower).

---

## Setup

From a terminal in the project root (`C:\videoAI`):

```bash
# 1. Create & activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS/Linux

# 2. Install PyTorch (CPU build) FIRST so pip doesn't pull the ~2.5 GB CUDA build
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

# 3. Install the rest
python -m pip install -r requirements.txt

# 4. Install & start the local LLM (Ollama)
#    (Ollama runs as a background service after install)
ollama pull qwen2.5:3b-instruct
```

On first run, the ASR model (~480 MB for `small`) and the embedding model
(~470 MB for `multilingual-e5-small`) are downloaded automatically into
`models/`.

---

## Running the app

```bash
venv\Scripts\activate
streamlit run app/streamlit_app.py
```

Then open the URL Streamlit prints (default <http://localhost:8501>).

**Tab 1 — 📥 Process Video**
1. Paste a video URL **or** upload a local file.
2. Click **Start**. A progress bar and live log show each stage
   (download → audio → transcribe → chunk → embed → ready).

**Tab 2 — 💬 Chat With Video**
1. Pick a processed video.
2. Ask a question. The answer appears with a **📍 References** expander listing
   the cited timestamp ranges, similarity scores, and the exact chunk text.

The sidebar shows Ollama connectivity, the active configuration, and the list of
processed videos.

---

## Command-line pipeline / tests

**End-to-end run** (download → … → ask questions), no UI:

```bash
# Persian defaults:
python tests/run_e2e.py --url "https://www.youtube.com/watch?v=<id>"

# Custom questions (repeat --q):
python tests/run_e2e.py --url "<url>" --q "این ویدیو درباره چیست؟" --q "نتیجه‌گیری چه بود؟"
```

**Unit tests** (fast, no models needed except imports):

```bash
python -m pytest tests/ -q
```

**Measure LLM VRAM** (requires Ollama running):

```bash
python tests/measure_vram.py
```

> On Windows, if you print Persian text to the console you may hit a `charmap`
> encoding error — that's a console limitation, not a pipeline failure. The test
> scripts set `PYTHONUTF8=1` / reconfigure stdout to avoid it; the Streamlit UI is
> unaffected.

---

## Configuration

Everything swappable lives in **`config/config.yaml`** — nothing is hardcoded in
the app. Highlights:

```yaml
asr:
  model_size: small        # tiny | base | small | medium  (NOT large-v3)
  device: auto             # auto -> CUDA if available, else CPU
  compute_type: int8       # int8 (cpu) | int8_float16/float16 (gpu)
  language: null           # null = auto-detect; "fa" to force Persian

embedding:
  model_name: intfloat/multilingual-e5-small   # multilingual; Persian-capable
  device: cpu

llm:
  backend: ollama
  model: qwen2.5:3b-instruct   # any Ollama tag; swappable
  keep_alive: "5m"             # "0" to unload from VRAM after each response

rag:
  top_k: 5
  min_score: 0.10          # drop chunks below this cosine similarity
```

**To enable GPU ASR** (optional, faster on long videos): install the CUDA runtime
libraries, then set `asr.device: cuda` and `asr.compute_type: int8_float16`:

```bash
python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

The transcriber registers those DLLs automatically and still falls back to CPU if
a GPU library error occurs at inference time.

---

## Project layout

```
config/config.yaml         Central config (models, paths, thresholds)
core/
  config.py                Typed config loader (dataclasses) + singleton
  utils.py                 Paths, logging, timestamp formatting, ids
  services.py              Wires db/embedder/llm/rag/pipeline together
modules/
  ingestion/downloader.py  yt-dlp download + local file / upload ingest
  audio/extractor.py       FFmpeg -> 16 kHz mono WAV; duration probe
  asr/transcriber.py       faster-whisper load/transcribe/unload, GPU->CPU fallback
  asr/chunker.py           Merge segments into ~30 s sentence-aware chunks
  embedding/embedder.py    sentence-transformers e5 (query/passage prefixes)
  vectorstore/            NumPy cosine store behind a VectorStore interface
  storage/db.py            SQLite: videos + chunks (+ embeddings as float32 BLOB)
  llm/ollama_client.py     Ollama HTTP client (proxy-bypassed), unload support
  rag/pipeline.py          Retrieve -> grounded answer with timestamp citations
  workflow/pipeline.py     Deterministic stage orchestrator (no agents)
app/streamlit_app.py       Two-tab UI: Process + Chat
tests/                     Unit tests, end-to-end driver, VRAM measurement
data/                      SQLite db, downloaded videos, audio, transcripts
models/                    Local cache for ASR + embedding weights
```

---

## Models used & why

| Role      | Model | Why |
|-----------|-------|-----|
| **ASR**   | `faster-whisper small` (int8) | Best quality/speed/VRAM trade-off on 4 GB. Multilingual incl. Persian. `large-v3` is intentionally disallowed (too heavy). Faster than real time on CPU. |
| **Embedding** | `intfloat/multilingual-e5-small` | Genuinely multilingual (strong Persian), small and CPU-friendly. Uses `query:` / `passage:` prefixes and L2-normalized vectors (cosine = dot product). |
| **LLM**   | `qwen2.5:3b-instruct` (Ollama) | Strong 3B instruct model with good multilingual/Persian ability that fits in ~2.1 GB VRAM. Swappable via config. |

**Persian ASR quality note:** the `small` model detects and transcribes Persian
well (language probability ≈ 0.97 in testing), but `small`/`int8` will make more
errors than a larger model on noisy audio, heavy accents, or domain jargon. Bump
`asr.model_size` to `medium` if you have the time/VRAM budget and need higher
accuracy.

---

## Known limitations

- **Aparat and some sites** can break as their players change — ingestion catches
  the error and reports a friendly message instead of crashing the app. Local
  upload and YouTube are the reliable paths.
- **ASR accuracy** is bounded by the `small` model (see note above).
- **Single-video chat**: retrieval is scoped to one selected video at a time.
- **No** speaker diarization, OCR, on-screen text, or vision — audio transcript
  only, by design for this MVP.
- **Vector store** is an in-memory NumPy index rebuilt per video from SQLite —
  perfect for MVP scale; swap in FAISS/Chroma behind the existing interface for
  large corpora.

See the final report (`REPORT.md`) for architecture rationale, test results, and
the future-upgrade roadmap.
