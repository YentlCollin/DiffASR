"""Import bridge for the versioned MawJais model implementation.

The model and processor live in two historical source directories.  Exposing
both directories through a real, top-level package keeps their relative imports
working and, unlike an entry created only in ``sys.modules``, lets spawn and
forkserver DataLoader workers import pickled processor classes in a fresh
interpreter.

Importing this package only establishes module search paths.  It deliberately
does not import Transformers modules, instantiate a processor, or load model
weights.
"""

from __future__ import annotations

from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
__path__ = [
    str(_REPOSITORY_ROOT / "audio" / "model"),
    str(_REPOSITORY_ROOT / "huggingface" / "xllm"),
]

__all__: tuple[str, ...] = ()
