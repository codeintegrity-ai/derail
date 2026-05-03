"""In-memory trace recorder for tests and hand-crafted traces.

**Not the production ingestion path.** Real scenario runs produce traces by
(a) running the agent via ``mcp-agent``, (b) collecting the OTEL spans it
emits, (c) reducing them through ``derail.metrics.process_spans`` into
``list[ToolCall]``.

``TraceRecorder`` exists for cases where spinning up ``mcp-agent`` + OTEL is
overkill — primarily unit tests that exercise evaluator logic against a
hand-crafted ``ToolCall`` sequence without the agent loop.

The recorder stamps each call with a monotonic ``seq`` and a ``transport``
label so the shape is indistinguishable from what ``process_spans`` produces.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from derail.metrics import ToolCall


@dataclass
class TraceRecorder:
    """Append-only collector of ``ToolCall`` records for one scenario run.

    ``transport`` labels every emitted call (e.g. ``"direct"`` or
    ``"direct+<classifier>"``) so the same trace shape can be produced by any
    transport without the evaluator needing to know which one ran.
    """

    transport: str
    redact: tuple[str, ...] = ()
    _trace: list[ToolCall] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _seq: int = 0

    def record(
        self,
        *,
        tool: str,
        args: dict[str, Any],
        result: Any,
        mcp_server: str | None = None,
        is_error: bool = False,
        error_message: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> ToolCall:
        """Record a completed tool call. Args matching ``redact`` are scrubbed."""
        scrubbed = {k: ("<redacted>" if k in self.redact else v) for k, v in args.items()}
        now = time.time()
        with self._lock:
            self._seq += 1
            call = ToolCall(
                tool=tool,
                args=scrubbed,
                result=result,
                start_time=start_time if start_time is not None else now,
                end_time=end_time if end_time is not None else now,
                seq=self._seq,
                transport=self.transport,
                mcp_server=mcp_server,
                is_error=is_error,
                error_message=error_message,
            )
            self._trace.append(call)
            return call

    def trace(self) -> list[ToolCall]:
        """Return a snapshot of the recorded trace (safe to iterate)."""
        with self._lock:
            return list(self._trace)

    def reset(self) -> None:
        with self._lock:
            self._trace.clear()
            self._seq = 0
