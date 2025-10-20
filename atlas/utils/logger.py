"""Logging utilities."""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = 'atlas', log_level: int = logging.INFO, 
                log_file: str = None) -> logging.Logger:
    """Setup logger with console and optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_session_log_file(log_dir: str = 'logs') -> str:
    """Create timestamped log file for session."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{log_dir}/atlas_{timestamp}.log"
