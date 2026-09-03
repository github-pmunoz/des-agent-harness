#!/usr/bin/env python3
"""
Watch for ESC keypresses in a separate thread. 
ESC-Cancel is posix-only; termios dep will raise on Windows.
"""

import os
import select
import sys
import termios
import threading
import tty


class ESCWatcher:
    """Watch for ESC keypresses in a separate thread to interrupt the LLM.
    The caller must pair start/stop with a try/finally block; the watcher does not 
    install atexit handlers by design. Prefer to use the context manager.
    ```
    with ESCWatcher() as esc_watcher:
    ```
    """
    ESC_GRACE = 0.05  # seconds to wait for the rest of an escape sequence

    def __init__(self):
        self.interrupted = False
        self.thread = None
        self.old_settings = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def start(self):
        if sys.stdin.isatty():
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()

    def _watch(self):
        fd = sys.stdin.fileno()
        while not self.interrupted:
            try:
                ready, _, _ = select.select([fd], [], [], 0.1)
                if ready:
                    char = os.read(fd, 1)
                    if char != b'\x1b':
                        continue  # type-ahead: consumed and dropped
                    # Bare ESC vs escape sequence: arrows/F-keys send "ESC [ ..." in one
                    # burst. If more bytes follow within the grace window it is a
                    # sequence -> drain and ignore. Silence means a real ESC keypress.
                    ready, _, _ = select.select([fd], [], [], self.ESC_GRACE)
                    if ready:
                        os.read(fd, 16)
                        continue
                    self.interrupted = True
                    break
            except OSError:
                break

    def stop(self):
        self.interrupted = True
        if self.thread:
            self.thread.join(timeout=1)
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
