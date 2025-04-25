import logging
import os
from contextlib import contextmanager

def setup_logging(loglevel=logging.INFO, logfile=None):
    logger = logging.getLogger()  
    logger.setLevel(loglevel)

    if not logger.handlers:
        formatter = logging.Formatter('%(levelname)s - %(message)s')

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if logfile:
            os.makedirs(os.path.dirname(logfile), exist_ok=True)
            file_handler = logging.FileHandler(logfile)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    if not hasattr(logger, "log_to_file"):
        def log_to_file_method(logfile, level=logging.INFO):
            @contextmanager
            def _context():
                fh = logging.FileHandler(logfile)
                fh.setLevel(level)
                formatter = logging.Formatter('%(levelname)s - %(message)s')
                fh.setFormatter(formatter)
                logger.addHandler(fh)
                try:
                    logger.debug(f"Logging to file: {logfile}")
                    yield
                finally:
                    logger.removeHandler(fh)
                    fh.close()
                    logger.debug(f"Stopped logging to file: {logfile}")
            return _context()
        logger.log_to_file = log_to_file_method

    return logger


logger = setup_logging(logging.DEBUG)

logger.info("HENDRICS logging initialized.")
