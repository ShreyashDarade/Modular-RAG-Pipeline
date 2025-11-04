import logging
from logging import Logger


def configure_logging(level: int = logging.INFO) -> Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("turinton-rag")


logger = configure_logging()
