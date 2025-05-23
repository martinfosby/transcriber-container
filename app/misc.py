from logger_config import get_logger, setup_logging

import logging
import sys

setup_logging(logging.INFO)
logger = get_logger(__name__)

def import_modules():
    logger.info("Importing required modules...")
    try:
        # Try to import required packages
        import importlib
        for package in ["aiofiles", "huggingface_hub", "transformers", "aiohttp"]:
            try:
                importlib.import_module(package)
            except ImportError:
                logger.warning(f"{package} not found, attempting to install...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logger.info(f"Successfully installed {package}")
    except Exception as e:
        logger.warning(f"Could not auto-install dependencies: {str(e)}")
