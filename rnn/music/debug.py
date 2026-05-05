# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------

OUTPUT = 1
ERROR  = 2
WARN   = 3
DEBUG  = 4
INFO   = 5

VERBOSITY = OUTPUT

def set_verbosity(level):
    global VERBOSITY
    VERBOSITY = level

def checkVerbosity(level):
    return VERBOSITY >= level

def info(*objects):
    if checkVerbosity(INFO):
        print(*objects)

def debug(*objects):
    if checkVerbosity(DEBUG):
        print(*objects)

def warn(*objects):
    if checkVerbosity(WARN):
        print(*objects)

def error(*objects):
    if checkVerbosity(ERROR):
        print(*objects)

def output(*objects):
    if checkVerbosity(OUTPUT):
        print(*objects)
