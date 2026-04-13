"""结构化日志"""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, stream=sys.stdout)
