#!/usr/bin/env python3
from enum import Enum

RAW_DATA_DIR = 'raw'
GRAPH_DIR = 'graphs'
DATA_DIR = 'data'


class Mode(Enum):
    PARSE = 1
    ANALYZE = 2
    ALL = 255

    def do_parse(self):
        return self == Mode.PARSE or self == Mode.ALL

    def do_analyze(self):
        return self == Mode.ANALYZE or self == Mode.ALL


class Type(Enum):
    NETEM = 1
    OPENSAND = 2

    @classmethod
    def from_name(cls, name):
        for t in cls:
            if t.name == name:
                return t
        return None