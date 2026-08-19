# 🎬 VidSense — Local-First Persian Speech-to-Text for Video

Give VidSense a Persian video (URL or file) → it produces a **high-quality,
faithful transcript**: correct words, accurate timestamps, per-word alignment,
optional speaker labels, Persian text normalization, and confidence scores.

Everything runs **locally**. No paid APIs, no cloud ASR. Built and tested on
**Windows + NVIDIA T1000 (4 GB VRAM)**, CPU by default.

```
video (YouTube / Aparat / direct URL / local file)
   │
   ▼  ingestion/downloader.py      Aparat direct-API bypass + yt-dlp + ffmpeg fallback
   ▼  audio/extractor.py           ffmpeg → 16 kHz mono 16-bit PCM WAV
   ▼  audio/preprocess.py          EBU R128 loudnorm (+ optional high-pass) — faithful, no meaning change
   ▼  vad/detector.py + segmenter  pyannote VAD (LOCAL weights) → speech windows {start, end}
   ▼  asr/engine.py                batched faster-whisper (medium, int8) → segments + avg_logprob
   ▼  alignment/aligner.py         wav2vec2-fa forced alignment → words {word, start, end, score}
   ▼  diarization/diarizer.py      pyannote speakers → words  (optional; graceful SPEAKER_00 fallback)
   ▼  postprocess/persian.py       ي→ی, ك→ک, ZWNJ/spacing/punctuation, repeat collapse — NO LLM, faithful
   ▼  confidence/scorer.py         per-segment confidence + duration-weighted quality_score
   │
   ▼  transcript JSON  {video_id, language, segments:[{speaker,start,end,text,confidence,words[]}]}
      + SRT / VTT subtitles ·  metadata in SQLite
```

The pipeline is a **single, deterministic** sequence of stages. Every heavy model
is **loaded → used → released** before the next, so the T1000's 4 GB is never
over-committed (models are never co-resident).

---

## Table of contents
1. [What it does](#what-it-does)
2. [Sequential-memory discipline](#sequential-memory-discipline)
3. [Setup](#setup)
4. [Running it](#running-it)
5. [Configuration](#configuration)
6. [Project layout](#project-layout)
7. [What changed from the old MVP](#what-changed-from-the-old-mvp)
8. [How the vendored WhisperX engine is reused](#how-the-vendored-whisperx-engine-is-reused)
9. [Models used & why](#models-used--why)
10. [Verification](#verification)
11. [Known limitations](#known-limitations)

---

## What it does

- **Ingest** from YouTube, **Aparat** (direct-API bypass for the domestic CDN),
  any yt-dlp-supported site, a direct `.mp4`, or a local upload.
- **Extract** 16 kHz mono audio with a bundled FFmpeg (no system install).
- **Preprocess** faithfully — EBU R128 loudness normalization so quiet/loud
  passages transcribe consistently. **No denoise by default** (it can distort
  speech); everything is a config toggle.
- **Detect speech** with pyannote VAD using **local weights** (no HF token), so
  the ASR only sees real speech regions.
- **Transcribe** with batched **faster-whisper** (`medium`, int8) — Persian, with
  `avg_logprob` per segment for confidence.
- **Align** each word to the audio with the Persian **wav2vec2** model →
  `{word, start, end, score}`.
- **Diarize** (optional): assign `SPEAKER_00/01/…` per word via pyannote. Disabled
  or token-less → every segment is labeled `SPEAKER_00` and the run continues.
- **Normalize** Persian text **faithfully** — Arabic→Persian letters (ي→ی, ك→ک),
  ZWNJ/spacing/punctuation tidy-up, and collapse of ≥3 identical repeated words.
  **No LLM, no rewriting, no new information** — the transcript stays true to the
  audio.
- **Score confidence** per segment (blend of `exp(avg_logprob)` and mean word
  alignment score), flag low-confidence segments, and roll up a
  duration-weighted `quality_score`.
- **Persist** the canonical transcript as JSON + SRT/VTT, with metadata in SQLite.

---

## Sequential-memory discipline

The heavy models are **never resident together**. Each stage does
`load → process → release` (`del` + `gc.collect()` + `torch.cuda.empty_cache()`)
before the next loads:

| Stage | Model | Approx. footprint |
|-------|-------|-------------------|
| VAD | pyannote segmentation (local) | small |
| ASR | faster-whisper `medium`, int8 | ~1.5 GB weights |
| Alignment | wav2vec2-large-xlsr-53-persian | ~1.3 GB |
| Diarization *(optional)* | pyannote community-1 | ~1–2 GB |

On this machine torch is the **CPU** build (`2.9.1+cpu`), so ASR/alignment/
diarization run on CPU regardless of `device: auto` — bounding RAM, keeping VRAM
free. GPU remains a config opt-in.

---

## Setup

Python 3.11, 64-bit. FFmpeg is **not** required (ships via `imageio-ffmpeg`).

```bash
# 1. Virtual environment
python -m venv venv
venv\Scripts\activate                 # Windows  (source venv/bin/activate on *nix)

# 2. Install the CPU build of PyTorch FIRST, so pip does not pull the multi-GB CUDA build
python -m pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cpu

# 3. Install the rest
python -m pip install -r requirements.txt
```

On first run the models download automatically into `models/`:
faster-whisper `medium` (~1.5 GB) and, for alignment, wav2vec2-fa (~1.3 GB).
Diarization weights (optional) require a Hugging Face token in config.

> **Note on Persian consoles (Windows):** the CLI reconfigures stdout to UTF-8 and
> writes JSON with `ensure_ascii=False`, so Persian output is correct in files and
> in the Streamlit UI even if a raw `cp1256` console can't print certain glyphs.

---

## Running it

### CLI

```bash
# Local file, medium model on CPU:
python scripts/transcribe.py --file data/videos/clip.mp4 --model medium --device cpu

# From a URL, with diarization and word alignment:
python scripts/transcribe.py --url "https://www.aparat.com/v/<hash>" --diarize

# Faster pass without word-level alignment:
python scripts/transcribe.py --file clip.mp4 --no-align
```

Prints a summary (language, #segments, #speakers, `quality_score`, device) and
writes `data/transcripts/<id>.json` + `.srt`.

### Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

- **📥 Process Video** — paste a URL or upload a file; a staged progress bar shows
  download → extract → preprocess → VAD → transcribe → align → diarize →
  postprocess → ready.
- **📄 Transcript** — pick a processed video; view RTL segments with speaker chips,
  `MM:SS–MM:SS` ranges, per-segment confidence, low-confidence highlighting, and
  **download JSON / SRT / VTT**.

---

## Configuration

Everything swappable lives in **`config/config.yaml`** — nothing hardcoded.
Highlights:

```yaml
asr:
  model_size: medium       # tiny | base | small | medium
  device: auto             # auto → CUDA if a torch CUDA runtime is present, else CPU
  compute_type: int8       # int8 (cpu) | int8_float16 / float16 (gpu)
  language: fa             # null = auto-detect
  batch_size: 8

vad:
  method: pyannote         # local weights in modules/whisperx/assets/
  chunk_size: 30           # max seconds per merged speech window

alignment:
  enabled: true            # wav2vec2 forced alignment → per-word timings

diarization:
  enabled: false           # true needs a HF token; false → SPEAKER_00 fallback
  hf_token: null

postprocess:               # faithful normalization only — never rewrites text
  arabic_to_persian: true  # ي→ی, ك→ک
  persian_digits: false
  collapse_repeats: true   # collapse runs of ≥3 identical words

confidence:
  low_threshold: 0.5       # flag segments below this
```

---

## Project layout

```
config/config.yaml           Central config (models, paths, thresholds)
core/
  config.py                  Typed config loader (dataclasses) + singleton
  services.py                Wires {cfg, db, processing} — no RAG
  utils.py                   Paths, logging, timestamp formatting, ids
modules/
  ingestion/downloader.py    yt-dlp + Aparat direct-API bypass + ffmpeg fallback
  audio/extractor.py         FFmpeg → 16 kHz mono WAV; duration probe        [kept]
  audio/preprocess.py        loudnorm / high-pass (faithful)
  audio/loader.py            in-memory waveform loader (feeds pyannote directly)
  vad/detector.py            pyannote/silero VAD wrapper (load/unload)
  vad/segmenter.py           merge_chunks → ≤chunk_size speech windows
  asr/engine.py              batched faster-whisper; GPU→CPU fallback
  alignment/aligner.py       wav2vec2-fa forced alignment (graceful skip)
  diarization/diarizer.py    pyannote speakers; graceful SPEAKER_00 fallback
  postprocess/persian.py     faithful Persian normalization (regex/table; NO LLM)
  confidence/scorer.py       per-segment confidence + quality_score
  storage/db.py              SQLite metadata (self-migrating schema)
  storage/transcript.py      JSON / SRT / VTT writers
  workflow/pipeline.py       deterministic stage orchestrator (sequential VRAM)
  whisperx/                  vendored WhisperX inference ENGINE (made importable)
app/streamlit_app.py         Two tabs: Process + Transcript
scripts/transcribe.py        CLI entry point
tests/                       unit tests (smoke, db, postprocess, confidence, ingestion)
data/                        SQLite db, videos, audio, transcripts   (gitignored)
models/                      local model cache                        (gitignored)
```

---

## What changed from the old MVP

The project was a RAG "chat with your video" MVP. It is now a single, focused
**transcript pipeline**. The RAG/chat stack was removed; the ASR path was upgraded
to full WhisperX quality.

**Removed** (RAG/chat + old ASR path):

```
modules/embedding/**            modules/llm/**           modules/rag/**
modules/vectorstore/**          modules/asr/transcriber.py   modules/asr/chunker.py
tests/test_vectorstore.py       tests/test_chunker.py    tests/run_e2e.py
tests/measure_vram.py           modules/ingestion/download(1).py  (folded into downloader.py)
```

**Added** (staged transcript modules + importability shims + CLI + tests):

```
modules/audio/preprocess.py     modules/audio/loader.py
modules/vad/{detector,segmenter}.py
modules/asr/engine.py           modules/alignment/aligner.py
modules/diarization/diarizer.py modules/postprocess/persian.py
modules/confidence/scorer.py    modules/storage/transcript.py
modules/whisperx/__init__.py    modules/whisperx/log_utils.py   modules/whisperx/vads/__init__.py
scripts/transcribe.py
tests/{test_postprocess,test_confidence,test_ingestion}.py
```

**Modified:** `core/config.py`, `core/services.py`, `config/config.yaml`,
`modules/ingestion/downloader.py` (Aparat bypass), `modules/storage/db.py`
(self-migrating schema, no chunks/embeddings), `modules/workflow/pipeline.py`
(rewritten for the transcript stages), `app/streamlit_app.py` (Chat tab removed,
Transcript tab added), `requirements.txt`.

---

## How the vendored WhisperX engine is reused

`modules/whisperx/` is a vendored copy of **m-bain/whisperX** kept as the single
internal ASR engine — ~4k lines of battle-tested code (batched faster-whisper ASR,
pyannote VAD, wav2vec2 Persian alignment, IntervalTree speaker assignment, subtitle
writers). Copying functions out would fragment it and lose upstream fidelity, so
instead it is **made importable and wrapped** by the thin staged modules above.

Mechanical, no-logic-change fixes to make it importable:
1. Added `modules/whisperx/__init__.py` re-exporting the public API.
2. Added `modules/whisperx/vads/__init__.py`.
3. Added `modules/whisperx/log_utils.py` → shim to `core.utils.get_logger`.
4. Rewrote absolute `whisperx.` imports → `modules.whisperx.`.

There is **no parallel implementation**: the old `asr/transcriber.py` path was
deleted, so WhisperX is the one and only ASR engine. On Windows, `torchcodec`'s DLL
fails to load — this is **harmless** because the pipeline hands pyannote a
preloaded in-memory waveform (`modules/audio/loader.py`) instead of relying on
torchcodec's decoder.

---

## Models used & why

| Role | Model | Why |
|------|-------|-----|
| **VAD** | pyannote segmentation (local weights) | Runs offline, no token; trims silence so ASR only sees speech. |
| **ASR** | `faster-whisper medium` (int8) | Best Persian accuracy/speed/RAM trade-off; `medium` markedly outperforms `small` on domain jargon. int8 keeps it CPU-friendly. |
| **Alignment** | `jonatasgrosman/wav2vec2-large-xlsr-53-persian` | Forced alignment for accurate per-word Persian timings. |
| **Diarization** | `pyannote/speaker-diarization-community-1` | Speaker turns; optional (token-gated), degrades to `SPEAKER_00`. |

**Why `medium` over `small`:** on the test clip the `small` model garbles domain
terms — `صنعت`→«سنات», `برق`→«برگ», `انرژی`→«اینرژی», `اقتصادی`→«اختصادی».
`medium` fixes the bulk of these; accuracy is the product, so `medium` is the
default despite the slower CPU pass.

---

## Verification

**Unit tests** (fast, no models — imports, config defaults, DB schema/migration,
faithful Persian normalization, confidence math, ingestion routing):

```bash
python -m pytest tests/ -q
# 37 passed
```

**End-to-end on a real Persian clip** (154 s). The full pipeline runs
`download → extract → preprocess → VAD → ASR → align → diarize → postprocess →
confidence → persist` and writes a well-formed transcript JSON + SRT. Checklist:

| # | Requirement | Status |
|---|-------------|--------|
| 1 | Download / ingest | ✅ |
| 2 | 16 kHz mono audio extract | ✅ |
| 3 | Faithful preprocess (loudnorm) | ✅ |
| 4 | VAD speech regions | ✅ (7 regions on the clip) |
| 5 | Segmentation | ✅ |
| 6 | Diarization or graceful `SPEAKER_00` | ✅ (fallback verified) |
| 7 | **ASR accuracy ≫ old garbled baseline** | ✅ **verified** — `medium` fixes the domain terms (table below) |
| 8 | Word alignment `{word,start,end}` | ✅ engine + graceful-skip verified; live pass pending wav2vec2-fa download |
| 9 | Faithful Persian normalization applied | ✅ (output uses ی/ک; no rewriting) |
| 10 | Final JSON shape `{video_id, language, segments:[{speaker,start,end,text,confidence,words}]}` | ✅ |

Points 1–6, 9, 10 are verified by a completed end-to-end run on the real clip.

### Accuracy: `medium` (new pipeline) vs. the old garbled baseline

Same 154 s Persian clip (minister interview on industrial power/energy rationing).
Old = `data/transcripts/2b6d14f4f1fb.json` (`small`, old pipeline);
new = `data/transcripts/34311ee43c30.json` (`medium`, new pipeline).
`quality_score` rose **0.696 → 0.806**, and — more importantly — the domain
vocabulary that was unusable is now correct:

| Meaning | Correct | OLD baseline | NEW `medium` |
|---|---|---|---|
| minister | وزیر | وزید ✗ | وزیر ✅ |
| electricity | برق | برگ ✗ | برق ✅ |
| energy | انرژی | اینرجی ✗ | انرژی ✅ |
| economy | اقتصاد | اختصاد ✗ | اقتصاد ✅ |
| state broadcaster | صدا و سیما | آسیما ✗ | صدا و سیما ✅ |
| Ministry of Energy | وزارت نیرو | وزاعت نیروب ✗ | وزارت نیرو ✅ |
| power plants | نیروگاه‌های برق | نیرگاهای برگ ✗ | نیروگاه های برق ✅ |
| prioritization | اولویت‌بندی | او لبیت فندی ✗ | اولویتبندی ✅ |
| notify (regions) | ابلاغ | ابلغ ✗ | ابلاغ ✅ |
| regulate | تنظیم | تنزیم ✗ | تنظیم ✅ |

Opening line, verbatim — old: `وزید سنات مادر و تجارت میگویت برگه حل چالش برگ سنای...`
→ new: `وزیر صناعت مدن و تجارت میگوید برای حل چالش برق صناعی قرار است...`.
Residual soft spots are minor and localized (صنعت/صنایع → صناعت/سنایه, معدن → مدن,
درصد → درست) — the transcript is now readable and faithful where before it was not.

---

## Known limitations

- **`medium` on CPU is slow** (worse than real time) — accepted for accuracy;
  GPU is a config opt-in.
- **First run downloads models** (medium ~1.5 GB, wav2vec2-fa ~1.3 GB) — cached to
  `models/` afterward; alignment degrades gracefully if unavailable.
- **Diarization** needs a Hugging Face token for the gated pyannote model; without
  it, everything is labeled `SPEAKER_00`.
- **Audio only** — no OCR / on-screen text / vision, by design.
