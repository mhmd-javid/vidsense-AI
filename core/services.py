"""Application service container — single place that wires the modules together.

Both the Streamlit app and the tests build their components through here so the
object graph (DB, embedder, LLM, RAG, processing pipeline) is consistent and
shares one warm CPU embedder.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.config import Config, load_config
from core.utils import get_logger, setup_logging
from modules.embedding.embedder import Embedder
from modules.llm.ollama_client import OllamaClient
from modules.rag.pipeline import RAGPipeline
from modules.storage.db import Database
from modules.workflow.pipeline import ProcessingPipeline

logger = get_logger(__name__)


@dataclass
class Services:
    cfg: Config
    db: Database
    embedder: Embedder
    llm: OllamaClient
    rag: RAGPipeline
    processing: ProcessingPipeline


def build_services(cfg: Config | None = None) -> Services:
    cfg = cfg or load_config()
    setup_logging(cfg.app.log_level)

    db = Database(cfg.db_path_abs)

    embedder = Embedder(
        model_name=cfg.embedding.model_name,
        device=cfg.embedding.device,
        batch_size=cfg.embedding.batch_size,
        normalize=cfg.embedding.normalize,
        query_prefix=cfg.embedding.query_prefix,
        passage_prefix=cfg.embedding.passage_prefix,
        cache_folder=str(cfg.models_dir_abs),
    )

    llm = OllamaClient(
        model=cfg.llm.model,
        host=cfg.llm.host,
        temperature=cfg.llm.temperature,
        num_ctx=cfg.llm.num_ctx,
        max_tokens=cfg.llm.max_tokens,
        keep_alive=cfg.llm.keep_alive,
        request_timeout=cfg.llm.request_timeout,
    )

    rag = RAGPipeline(
        db=db,
        embedder=embedder,
        llm=llm,
        backend=cfg.vectorstore.backend,
        top_k=cfg.rag.top_k,
        min_score=cfg.rag.min_score,
    )

    processing = ProcessingPipeline(cfg=cfg, db=db, embedder=embedder)

    return Services(cfg=cfg, db=db, embedder=embedder, llm=llm, rag=rag, processing=processing)
