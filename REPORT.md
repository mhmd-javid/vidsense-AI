# VideoAI — Final Report

**A local-first AI Video Intelligence MVP.** User provides a video → the system
extracts the knowledge inside it → the user chats with the video and receives
answers grounded in the transcript, each citing the timestamps it came from.

Everything runs locally on the target hardware (Windows, **NVIDIA T1000, 4 GB
VRAM**). No paid APIs; ASR, embeddings, and the LLM are all local.

---

## 1. What was built

A complete, working pipeline plus a two-tab Streamlit UI:

1. **Ingestion** — download by URL via `yt-dlp` (YouTube reliable; Aparat/others
   best-effort with graceful failure) or accept a local file / upload.
2. **Audio extraction** — a bundled static FFmpeg (`imageio-ffmpeg`) produces
   16 kHz mono WAV; duration probed via PyAV.
3. **ASR** — `faster-whisper` (CTranslate2) transcribes to segments with
   timestamps + detected language + language probability.
4. **Chunking** — many tiny ASR segments are merged into ~30 s, sentence-aware
   chunks with accurate preserved `start`/`end`.
5. **Embedding** — `intfloat/multilingual-e5-small` (Persian-capable),
   L2-normalized, CPU.
6. **Storage** — SQLite: `videos` + `chunks` (embeddings stored as float32 BLOBs);
   schema is PostgreSQL-migratable.
7. **RAG chat** — retrieve top-k chunks by cosine similarity, answer **only** from
   them, **always cite timestamps** (`[MM:SS–MM:SS]`), refuse when unsupported.
8. **Orchestration** — a deterministic stage runner (`download → extract →
   transcribe → chunk → embed → ready`) with progress callbacks. No agents, no
   LangGraph — but every module sits behind a clean interface, so an agent layer
   is a drop-in later.

**Status: working end-to-end.** A Persian video processes fully and the chat
returns a grounded Persian answer with a correct timestamp citation.

---

## 2. Architecture

```
┌────────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────┐
│ Ingestion  │──►│  Audio   │──►│ ASR (Whisper)│──►│ Chunker  │
│ yt-dlp/file│   │ FFmpeg   │   │ CPU int8     │   │ ~30s      │
└────────────┘   └──────────┘   └──────────────┘   └────┬─────┘
                                                        │
        ┌───────────────────────────────────────────────┘
        ▼
┌──────────────┐   ┌──────────────────┐        ┌───────────────────────┐
│ Embedder     │──►│ SQLite + Vector  │◄──────►│ RAG: retrieve + ground │
│ e5 (CPU)     │   │ store (per video)│        │ + timestamp citations  │
└──────────────┘   └──────────────────┘        └───────────┬───────────┘
                                                            ▼
                                          ┌──────────────────────────────┐
                                          │ LLM (Ollama, GPU) qwen2.5:3b  │
                                          └──────────────────────────────┘
```

**Design principles**

- **Config-driven, nothing hardcoded** — models, devices, thresholds, paths all
  live in `config/config.yaml`, loaded into typed dataclasses (`core/config.py`).
- **Sequential model residency** — the whole reason ASR and embeddings run on CPU
  is to guarantee they never share VRAM with the LLM. On 4 GB that turns a
  contention problem into a non-problem.
- **Interfaces over implementations** — `VectorStore` is abstract (NumPy today,
  FAISS/Chroma later); the LLM client and ASR/embedder each expose
  `load()`/`unload()`; the workflow emits `(stage, message, fraction)` callbacks.
- **Fail soft** — ingestion and the whole workflow catch errors, mark the video
  `ERROR` with a message, and never crash the UI.
- **Grounded generation** — the RAG system prompt forbids answering outside the
  retrieved excerpts and mandates timestamp citations.

**Module map**: `core/` (config, utils, service wiring) · `modules/`
(`ingestion`, `audio`, `asr`, `embedding`, `vectorstore`, `storage`, `llm`,
`rag`, `workflow`) · `app/` (Streamlit) · `tests/` · `config/` · `data/` ·
`models/`.

---

## 3. Dependencies

Installed **only what the project needed**; nothing pre-existing was reinstalled.

| Package | Version | Role |
|---|---|---|
| torch (CPU build) | 2.13.0+cpu | backs sentence-transformers on CPU |
| faster-whisper | 1.2.1 | ASR |
| ctranslate2 | 4.8.1 | Whisper inference backend |
| sentence-transformers | 3.4.1 | embeddings |
| transformers / tokenizers / huggingface_hub | 4.57.6 / 0.22.2 / 0.36.2 | model loading |
| yt-dlp | 2026.7.4 | video download |
| imageio-ffmpeg | 0.6.0 | bundled FFmpeg binary |
| av (PyAV) | 18.1.0 | duration probe |
| numpy | 2.4.6 | vector math / storage |
| scikit-learn | 1.9.0 | cosine helper |
| httpx | 0.28.1 | Ollama HTTP client |
| streamlit | 1.61.1 | UI |
| PyYAML | 6.0.3 | config |

**Key install decision:** PyTorch is installed from the **CPU wheel index first**,
avoiding the ~2.5 GB CUDA build we don't need (embeddings run on CPU). Ollama is
an external service (installed separately, not a pip dependency).

---

## 4. Models chosen — and why

| Role | Model | Rationale |
|---|---|---|
| ASR | `faster-whisper small`, `int8` | Best quality/speed/VRAM balance for 4 GB. Multilingual incl. Persian. `large-v3` deliberately disallowed. Runs **faster than real time on CPU** (RTF ≈ 0.9). |
| Embedding | `intfloat/multilingual-e5-small` | Truly multilingual with strong **Persian** coverage (not English-only). Small, CPU-friendly. Requires `query:`/`passage:` prefixes; vectors L2-normalized so cosine = dot product. |
| LLM | `qwen2.5:3b-instruct` via Ollama | Strong ~3B instruct model with good Persian/multilingual ability, fits ~2.1 GB VRAM. Config-swappable to any Ollama tag. |

---

## 5. How to run

```bash
python -m venv venv && venv\Scripts\activate
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
ollama pull qwen2.5:3b-instruct

streamlit run app/streamlit_app.py          # UI
python tests/run_e2e.py --url "<video-url>"  # headless end-to-end
python -m pytest tests/ -q                   # unit tests
```

Full details in `README.md`.

---

## 6. Test results

**Unit tests — `pytest tests/`: `20 passed`.**

- `test_chunker.py` (5) — merging short segments into target windows; hard caps on
  duration **and** characters; sentence-boundary preference; empty input. These
  caught two real chunker bugs during development (cap checked after append;
  trailing merge-back overriding boundaries) — both fixed.
- `test_vectorstore.py` (5) — cosine ranking, metadata (`video_id`) filtering,
  upsert-replace, delete, count.
- `test_db.py` (5) — upsert/get, status transitions, float32 embedding round-trip,
  idempotent `replace_chunks`, cascade delete.
- `test_smoke.py` (5) — all modules import, config loads with expected defaults,
  timestamp formatting, bundled FFmpeg present, vector-store factory.

**Streamlit UI** — verified in-process with `streamlit.testing` `AppTest`: the
script runs top-to-bottom **with no exception**, both tabs render, services load,
and Ollama status is queried successfully.

**End-to-end** — a Persian video runs through the full pipeline (download → audio
→ ASR → chunk → embed → ready) and the chat returns a **grounded Persian answer
with a correct timestamp citation**. Persian ASR language probability ≈ **0.97**.

---

## 7. VRAM observations (NVIDIA T1000, 4 GB)

Measured with `tests/measure_vram.py` (LLM isolated: unload → baseline → chat):

```
Baseline (desktop, no LLM):   888 / 4096 MiB
LLM (qwen2.5:3b) loaded    :  2994 / 4096 MiB
------------------------------------------------
LLM footprint              ≈ 2106 MiB
Headroom                   ≈ 1102 MiB
```

- **ASR and embeddings contribute 0 MiB** (CPU) — so the LLM is the only GPU
  consumer and there is **never** three-model co-residency.
- ~1.1 GB headroom remains with the LLM loaded — comfortable for `qwen2.5:3b` and
  leaving room for GPU ASR later if desired.

---

## 8. Environment issues encountered & resolved

1. **CUDA cuBLAS/cuDNN DLLs missing** → GPU ASR failed at inference. **Resolved**
   by auto-detecting the GPU library error and falling back to CPU int8 (works,
   faster than real time). GPU ASR is now an opt-in (see §10 / config).
2. **Ollama "server disconnected"** → `httpx` was routing `localhost:11434`
   through a system (WARP) proxy. **Resolved** with `trust_env=False`.
3. **Windows console `charmap` error** printing Persian → console-only, not a
   pipeline failure. **Resolved** in scripts with UTF-8 stdout; UI unaffected.

---

## 9. Known limitations

- **ASR accuracy** bounded by `small`/`int8` — more errors on noisy audio, strong
  accents, or jargon. Bump to `medium` for higher accuracy at higher cost.
- **Aparat / non-YouTube sites** may break as players change; handled gracefully
  (friendly error, no crash). YouTube and local upload are the reliable paths.
- **Single-video chat** — retrieval is scoped to one video at a time.
- **No** diarization, OCR/on-screen text, or vision — audio transcript only (by
  design for this MVP).
- **Vector store** is an in-memory NumPy index rebuilt per video from SQLite —
  ideal at MVP scale, not for very large corpora.

---

## 10. Future upgrades

- **GPU ASR opt-in**: `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`, set
  `asr.device: cuda`, `compute_type: int8_float16` — faster on long videos, still
  well within the VRAM budget since it loads before the LLM.
- **Scale the vector store**: swap NumPy for **FAISS/Chroma** behind the existing
  `VectorStore` interface; enable multi-video / whole-library search.
- **Storage**: migrate SQLite → **PostgreSQL** (schema already compatible) +
  `pgvector`.
- **Richer extraction**: speaker **diarization**, **OCR / on-screen text**, and
  **vision** captioning to capture non-spoken content.
- **Agent layer**: the deterministic pipeline's interfaces are agent-ready — add
  planning/tool-use (e.g. LangGraph) without rewriting the modules.
- **Quality**: larger ASR (`medium`/distil), reranking retrieved chunks,
  long-video summarization and chaptering.
- **Ops**: batch/queue processing, progress persistence, and basic monitoring
  (latency, RTF, VRAM) for longer-running jobs.

---

## Appendix — quick facts

- **Python** 3.11.9 · Windows 11 · NVIDIA T1000 4 GB
- **ASR** faster-whisper `small`/int8 on CPU · **Embed** e5-small on CPU ·
  **LLM** qwen2.5:3b-instruct on GPU via Ollama
- **Tests**: 20/20 unit passing; UI + end-to-end verified
- **LLM VRAM** ≈ 2.1 GB; ASR/embeddings 0 GPU by design
