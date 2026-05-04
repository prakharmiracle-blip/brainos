"""
BrainOS — Centralized Logger
"""

import sys
from loguru import logger
from src.config import settings


def setup_logger():
    logger.remove()  # remove default handler

    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
        colorize=True,
    )

    logger.add(
        "logs/brainos.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
    )

    return logger


# Initialise once and export
import os
os.makedirs("logs", exist_ok=True)
log = setup_logger()
