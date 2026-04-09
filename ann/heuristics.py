from argParser import *
from debug import *

class Heuristics:
    def __init__(self):
        self.epochData = dict()
        self.finalData = dict()
        self.isTraining = True

    def dumpConfig(self):
        items = " ".join(
            f"{key}={value}" for key, value in sorted(vars(args).items())
        )
        output(f"CONFIG {items}")

    def dumpEpoch(self):
        items = " ".join(
            f"{key}={value}" for key, value in sorted(self.epochData.items())
        )
        output(f"EPOCH {items}")

    def dumpFinal(self):
        items = " ".join(
            f"{key}={value}" for key, value in sorted(self.finalData.items())
        )
        output(f"FINAL {items}")

heuristics = Heuristics()
