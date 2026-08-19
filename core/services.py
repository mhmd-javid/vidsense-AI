"""Application service container — single place that wires modules together.

Both the Streamlit app and the CLI build their components through here so the
object graph (config, DB, processing pipeline) is consistent.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.config import Config, load_config
from core.utils import get_logger, setup_logging
from modules.storage.db import Database
from modules.workflow.pipeline import ProcessingPipeline

logger = get_logger(__name__)


@dataclass
class Services:
    cfg: Config
    db: Database
    processing: ProcessingPipeline


def build_services(cfg: Config | None = None) -> Services:
    cfg = cfg or load_config()
    setup_logging(cfg.app.log_level)

    db = Database(cfg.db_path_abs)
    processing = ProcessingPipeline(cfg=cfg, db=db)

    return Services(cfg=cfg, db=db, processing=processing)
