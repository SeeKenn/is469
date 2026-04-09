"""
seeding.py
Utilities for applying the configured project seed consistently across
scripts, evaluation runs, and interactive app sessions.
"""

import os
import random
from typing import Optional

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def set_global_seed(seed: int, deterministic: bool = True) -> int:
    """
    Apply the same seed to common Python and ML randomness sources.

    Notes:
    - PYTHONHASHSEED only affects hash randomization for new processes, but
      setting it here helps subprocesses inherit the configured value.
    - Some third-party libraries may still use nondeterministic operations.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except TypeError:
                    torch.use_deterministic_algorithms(True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logger.info(f"Applied global random seed: {seed}")
    return seed


def seed_from_config(cfg: Optional[dict] = None, deterministic: bool = True) -> Optional[int]:
    """
    Read the project seed from config/settings.yaml and apply it if present.
    """
    cfg = cfg or load_config()
    seed = cfg.get("project", {}).get("seed")
    if seed is None:
        logger.info("No project seed configured; skipping seed setup")
        return None
    return set_global_seed(int(seed), deterministic=deterministic)
