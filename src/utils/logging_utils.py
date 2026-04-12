from __future__ import annotations

import csv
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf
from rich.logging import RichHandler


class ExperimentLogger:
    def __init__(self, pylogger: logging.Logger, csv_path: Path, wandb_run=None):
        self._logger = pylogger
        self.csv_path = csv_path
        self.wandb_run = wandb_run

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def log(self, metrics_dict: dict, step: int | None = None):
        if metrics_dict is None:
            return

        stamp = datetime.utcnow().isoformat()
        step_val = -1 if step is None else int(step)

        self._logger.info("step=%s metrics=%s", step_val, metrics_dict)

        with self.csv_path.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            for k, v in metrics_dict.items():
                writer.writerow([stamp, step_val, k, v])

        if self.wandb_run is not None:
            try:
                self.wandb_run.log(metrics_dict, step=step)
            except Exception as exc:
                self._logger.warning("wandb_log_failed err=%s", str(exc))


def config_hash(config) -> str:
    payload = OmegaConf.to_container(config, resolve=True)
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def setup_logger(config) -> ExperimentLogger:
    """
    Rich console logger with CSV metric persistence and optional wandb.
    """
    results_dir = Path(config.logging.results_dir)
    metrics_dir = Path(config.logging.metrics_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    pylogger = logging.getLogger("nifty100")
    pylogger.setLevel(logging.INFO)
    pylogger.handlers = []
    handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
    fmt = logging.Formatter("%(message)s")
    handler.setFormatter(fmt)
    pylogger.addHandler(handler)
    pylogger.propagate = False

    csv_path = metrics_dir / "run_log.csv"
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["timestamp", "step", "metric", "value"])

    wandb_run = None
    if bool(config.logging.use_wandb):
        try:
            import wandb

            wandb_run = wandb.init(
                project=str(config.logging.project_name),
                config=OmegaConf.to_container(config, resolve=True),
                reinit=True,
            )
            pylogger.info("wandb_initialized project=%s", str(config.logging.project_name))
        except Exception as exc:
            pylogger.warning("wandb_unavailable fallback_to_csv err=%s", str(exc))

    logger = ExperimentLogger(pylogger=pylogger, csv_path=csv_path, wandb_run=wandb_run)
    logger.info("config_hash=%s", config_hash(config))
    logger.info("start_timestamp=%s", datetime.utcnow().isoformat())
    return logger
