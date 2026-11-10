"""Importable forkserver target used by the repository-loader regression test."""

from __future__ import annotations

import inspect
import pickle
import traceback


def unpickle_processor_class(serialized_class: bytes, connection) -> None:
    """Unpickle one class in a fresh worker and return its import provenance."""

    try:
        processor_class = pickle.loads(serialized_class)
        connection.send(
            {
                "ok": True,
                "module": processor_class.__module__,
                "name": processor_class.__name__,
                "source": inspect.getsourcefile(processor_class),
            }
        )
    except BaseException:  # pragma: no cover - parent reports the child traceback
        connection.send({"ok": False, "traceback": traceback.format_exc()})
    finally:
        connection.close()
