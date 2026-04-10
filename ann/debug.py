# --------------------------------------------------
# Tech Aarvam
# Copyright (c) 2026 Tech Aarvam.
# Author: Ram (Ramasubramanian B)
# --------------------------------------------------

VERBOSITY=1

def info(*objects):
    if VERBOSITY >=5:
        print (*objects)

def debug(*objects):
    if VERBOSITY >=4:
        print (*objects)

def warn(*objects):
    if VERBOSITY >=3:
        print (*objects)

def error(*objects):
    if VERBOSITY >=2:
        print (*objects)

def output(*objects):
    if VERBOSITY >=1:
        print (*objects)
