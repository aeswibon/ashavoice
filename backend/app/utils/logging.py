import logging

from app.utils.config import settings


def setup_logging():
    """
    Sets up a centralized logging configuration for the application.
    Logs will be written to a file and also output to the console.
    """

    # Get log level from environment variable, default to INFO
    log_level = getattr(logging, settings.LOG_LEVEL, logging.INFO)

    # Create a custom logger
    logger = logging.getLogger("asha_voice_logger")
    logger.setLevel(log_level)

    # Prevent duplicate handlers if setup_logging is called multiple times
    if not logger.handlers:
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # File Handler: writes logs to a file
        file_handler = logging.FileHandler(settings.LOG_FILE_NAME)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console Handler: outputs logs to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


# Initialize the logger immediately when this module is imported
# This ensures that any module importing this will get a configured logger
logger = setup_logging()

# You can add a shutdown hook if necessary for file handlers, though often not critical for small apps
# import atexit
# atexit.register(logging.shutdown)
