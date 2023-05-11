# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# TODO: Add description
# ---------------------------------------------------------
from sys import argv
from util.env import RUN_PATH
from lib.Signal import Signal


LOG = RUN_PATH / "log.txt"
LOG.touch(exist_ok=True)
LOG_BK = RUN_PATH / ".log.txt"
LOG_BK.touch(exist_ok=True)


def add(id: str):
    with open(LOG, "a") as f:
        print(f"{id} | {' '.join(argv)}", file=f)


def remove(id: str, reason: str = "interrupted"):
    with Signal(abortable=False):
        with open(LOG, "r") as f:
            lines = f.readlines()
        with open(LOG, "w") as log, open(LOG_BK, "a") as bk:
            for line in lines:
                if line.startswith(id):
                    bk.write(f"{reason}: {line}")
                else:
                    log.write(line)
