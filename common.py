#!/usr/bin/env python3
import logging
import sys
from enum import Enum

RAW_DATA_DIR = 'parsed'
GRAPH_DIR = 'graphs'
DATA_DIR = 'data'

TYPE_FILE = '.type'
AUTO_DETECT_FILE = '.auto-detect'

logger = logging.getLogger(__name__)


def setup_logger():
    global logger

    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s %(processName)-12s [%(levelname)s] %(message)s'))
    logger.addHandler(handler)


class Mode(Enum):
    PARSE = 1
    ANALYZE = 2
    ALL = 255

    def do_parse(self):
        return self == Mode.PARSE or self == Mode.ALL

    def do_analyze(self):
        return self == Mode.ANALYZE or self == Mode.ALL


class MeasureType(Enum):
    NETEM = 1
    OPENSAND = 2

    @classmethod
    def from_name(cls, name):
        for t in cls:
            if t.name == name:
                return t
        return None
