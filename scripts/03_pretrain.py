"""
Usage: python scripts/03_pretrain.py

1. Load config
2. Setup logger
3. Run src.training.pretrain.run_pretraining(config)
4. Plot pretrain loss curve
5. Log: pretrain complete, checkpoint path
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.pretrain import run_pretraining
from src.utils.logging_utils import setup_logger


def main():
    t0 = time.time()
    config = OmegaConf.load(PROJECT_ROOT / "config" / "config.yaml")
    logger = setup_logger(config)
    out = run_pretraining(config, logger=logger)
    logger.info("pretraining_done checkpoint=%s best_loss=%.6f", out["checkpoint_path"], out["best_loss"])
    logger.info("end_timestamp=%s runtime_sec=%.2f", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), time.time() - t0)


if __name__ == "__main__":
    main()
