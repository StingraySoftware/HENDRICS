import logging
import os

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

    return logger


logger = setup_logging(logging.DEBUG)

logger.info("HENDRICS logging initialized.")
