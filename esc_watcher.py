import sys
import os
import threading
import select

class ESCWatcher:
    def __init__(self):
        self.interrupted = False
        self.thread = None
        self.old_settings = None
        self.is_unix = os.name == 'posix'

    def start(self):
        if self.is_unix:
            import termios
            import tty
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()

    def _watch(self):
        if self.is_unix:
            fd = sys.stdin.fileno()
            while not self.interrupted:
                try:
                    ready, _, _ = select.select([fd], [], [], 0.1)
                    if ready:
                        char = os.read(fd, 1)
                        if char == b'\x1b':
                            self.interrupted = True
                            break
                except OSError:
                    break
        else:
            import time
            while not self.interrupted:
                time.sleep(0.1)

    def stop(self):
        self.interrupted = True
        if self.thread:
            self.thread.join(timeout=1)
        if self.is_unix and self.old_settings:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)