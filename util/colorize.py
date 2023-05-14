# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# TODO: Add description
# ---------------------------------------------------------
# PIP Modules
from termcolor import colored
from sys import stdout, stderr

ENABLED = stderr.isatty()

SCHEME = [
    "black",
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
    "white",
    "light_grey",
    "dark_grey",
    "light_red",
    "light_green",
    "light_yellow",
    "light_blue",
    "light_magenta",
    "light_cyan",
]


def printer(color):
    def with_color(*args):
        return [colored(arg, color=color) for arg in args]

    return with_color


class Colorizer:
    def __init__(self, file=stdout) -> None:
        def plain(*args):
            return args

        for color in SCHEME:
            setattr(self, color, printer(color) if file.isatty() else plain)


colorize = Colorizer()
