# logger_config.py

import logging

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
