import asyncio
import logging
import sys

from app.config import settings
from app.transcriber import run_worker


def configure_logging() -> None:
    level_name = getattr(settings, "LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


if __name__ == "__main__":
    configure_logging()
    asyncio.run(run_worker())

