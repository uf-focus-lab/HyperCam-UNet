# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# Context manager for running a model or a task in a run.
# ---------------------------------------------------------
# Internal Modules
import traceback
from sys import stderr
from pathlib import Path
from os import scandir, mkdir
from os.path import isdir, basename
from math import floor, ceil
from datetime import datetime
from shutil import rmtree

# PIP Modules
import torch

# Custom Modules
from lib.Signal import Signal
import util.run_log as run_log
from util.colorize import colorize
from util.env import RUN_PATH, ensure, relative


def getRunID(start_i: int = 0):
    now = datetime.now().strftime("%Y%m%d")[2:]
    i = start_i
    while True:
        name = f"{now}-{i:02d}"
        if not isdir(RUN_PATH / name):
            mkdir(RUN_PATH / name)
            return name
        else:
            i += 1


def getRunList():
    return [basename(_.path) for _ in scandir(RUN_PATH) if _.is_dir()]


class Runtime:
    epochs: int = None
    train_mode: list[str] = None
    optimizer: torch.nn.Module = None
    sample_size: int = 5

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class Context(Runtime):
    def __init__(self, id, path, parent=None, **kwargs):
        super().__init__(**kwargs)
        ensure(path)
        self.id = id
        self.path: Path = path
        self.parent = parent
        self.signal = Signal(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def log(self, *args, file="log.txt", banner=None, visible=True):
        if len(args) == 0 and banner is None:

            def p(*args):
                if len(args) == 0:
                    return
                self.log(*args, file=file)

            return p
        # Print banner (optional)
        if banner is not None:
            w = (78 - len(banner)) / 2
            print()
            if w < 1:
                print(*colorize.dark_grey(banner))
            else:
                l = "".join(["="] * ceil(w))
                r = "".join(["="] * floor(w))
                print(*colorize.dark_grey(l, banner, r))
        # Print args if applicable
        if len(args):
            if isdir(self.path):
                with open(self.path / file, "a") as log:
                    print(*args, file=log)
            else:
                print("# Unable to reach", self.path, file=stderr)
            # Duplex to stdout
            if visible:
                print(*colorize.cyan(self.id), *colorize.dark_grey("|"), *colorize.light_grey(*args))

    def interrupt(self, code: int = -1):
        if self.parent is not None:
            self.parent.interrupt(code)

    __memory__ = {}

    def push(self, key, *values):
        """
        Push list of values into temp memory of current context.
        """
        if key in self.__memory__:
            assert isinstance(self.__memory__[key], list)
            for el in values:
                self.__memory__[key].append(el)
        else:
            self.__memory__[key] = [el for el in values]

    def collect(self, key, clear=True) -> list:
        """
        Collect all values previously pushed into memory.
        Removes collected set of value by default.
        """
        assert key in self.__memory__, key
        mem = self.__memory__[key]
        if clear:
            del self.__memory__[key]
        return mem

    def collect_all(self, clear=True):
        def iterate_key(key: str):
            return key, self.collect(key, clear=clear)

        return map(iterate_key, list(self.__memory__.keys()))


class Run(Context):
    state = None

    def __init__(self, id=None):
        if id is None:
            id = getRunID()
        super().__init__(id, RUN_PATH / id)
        self.signal.__enter__()
        self.signal.triggered = True
        self.log(banner=f"RUN ID: {id}")
        run_log.add(id)

    def context(self, context_name="train", **kwargs):
        return Context(self.id, self.path / context_name, self, **kwargs)

    def interrupt(self, code: int = -1):
        self.log("")
        self.log(banner="Interrupted")
        self.state = code
        rmtree(self.path)
        run_log.remove(self.id)

    def __exit__(self, errType, err, trace):
        if self.state is None and errType is None:
            now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            self.log(now, file="000_SUCCESS", banner="Finished")
            self.signal.__exit__()
        elif errType is not None and errType != SystemExit:
            # Filter file name from traceback and replace with relative path
            print(file=stderr)
            print(*colorize.yellow("Traceback:"), file=stderr)
            print(*colorize.yellow("=========="), file=stderr)
            for tb in traceback.extract_tb(trace):
                tb.filename = str(relative(tb.filename))
                # Print traceback
                print(*colorize.dark_grey(f"  {tb.filename}:{tb.lineno}", tb.name), file=stderr)
                print(*colorize.yellow("  " + tb.line.strip()), file=stderr)
            try:
                rmtree(self.path)
            except:
                pass
            print("\n", *colorize.red(str(err).strip()), "\n", file=stderr)
            run_log.remove(self.id, "aborted on error")
        return True


if __name__ == "__main__":
    with Run() as run:
        pass
