"""
Usage: python scripts/01_download_data.py

Loads config, calls src.data.download.download_all(), logs progress.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download import download_all
from src.utils.logging_utils import setup_logger


def main():
    t0 = time.time()
    config = OmegaConf.load(PROJECT_ROOT / "config" / "config.yaml")
    logger = setup_logger(config)
    summary = download_all(config, logger=logger)
    logger.info(
        "download_summary requested=%d downloaded=%d skipped=%d missing=%d",
        summary.requested,
        summary.downloaded,
        summary.skipped_fresh,
        summary.missing,
    )
    logger.info("end_timestamp=%s runtime_sec=%.2f", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), time.time() - t0)


if __name__ == "__main__":
    main()
