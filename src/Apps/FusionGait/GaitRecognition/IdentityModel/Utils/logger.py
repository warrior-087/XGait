import logging
import os
from datetime import datetime


def get_logger(name, save_path=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding duplicate handlers
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    # Console output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File output
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file_handler = logging.FileHandler(save_path, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info(f"Logger initialized. Log file: {save_path}")
    return logger
