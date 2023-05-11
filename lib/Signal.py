# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# Registers handler for a specific signal, compatible with
# the python "with" statement for scoped capturing.
# ---------------------------------------------------------
from signal import signal, getsignal, SIGINT
from sys import exit


class Signal:
    triggered = False
    abortable = True

    def handler(self, signal, *args, **kwargs):
        assert signal == self.signal, signal
        if not self.abortable:
            print("\nSignal ignored: current procedure is not abortable.")
            return
        if self.triggered:
            if self.context is not None:
                self.context.interrupt()
            exit(1)
        else:
            self.triggered = True
            print("")
            if self.context is not None:
                self.context.log("[SIGNAL Triggered] Aborting...")
            else:
                print("[SIGNAL Triggered] Aborting...")
            print("Press Ctrl-C again to exit NOW")

    def __init__(self, context=None, sig=SIGINT, abortable=True):
        self.signal = sig
        self.context = context
        self.abortable = abortable

    def __enter__(self):
        self.previous_handler = getsignal(self.signal)
        signal(self.signal, self.handler)
        return self

    def __exit__(self, *args):
        signal(self.signal, self.previous_handler)
        self.triggered = None


if __name__ == "__main__":
    with Signal() as sig:
        print("Capturing, press CTRL-C to exit capture ...")
        while not sig.triggered:
            pass
    print("Exiting Capture, press CTRL-C again ...")
    while True:
        pass
