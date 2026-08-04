"""Process-wide synchronization for PDFium's non-thread-safe boundary."""

from __future__ import annotations

import threading


# BatchRunner uses worker threads. PDFium supports separate processes, but its
# Python bindings do not guarantee concurrent calls from threads in one
# process. Every module that opens or renders a PDF shares this re-entrant lock.
PDFIUM_RENDER_LOCK = threading.RLock()
