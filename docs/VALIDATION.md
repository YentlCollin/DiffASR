# Public release validation — 10 November 2026

The public-source CPU suite passed: **189 tests**. Three modules that require
withheld base-model files are excluded from collection while those files are
absent; see [INTEGRATION.md](INTEGRATION.md).

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
python -m pytest -q -p no:cacheprovider
```

Validation used Python 3.11 on macOS with a CPU test environment. The pinned
Transformers, Accelerate, PEFT and PyArrow versions in `requirements.txt` were
installed for the final run. The GPU trainer still requires the supported
ROCm environment; CPU tests do not establish GPU training support on macOS.

Additional checks:

- All Python source files parse; shell launchers pass `bash -n`.
- The Slurm submission dry-run handles paths with spaces and creates no outputs.
  Path resolution uses Python rather than GNU-only `realpath` options.
- A detect-secrets 1.5.0 scan and targeted checks found no embedded access
  credentials or retained internal model/dataset repository identifiers.
- The public snapshot contains no base-model implementations, weights, datasets,
  tokenizer assets, private model patch, or original Git history.

No full-model inference, GPU training, real-checkpoint export, full-corpus
preprocessing, or recognition-quality evaluation was performed for this release.

`python tools/verify_bundle.py` verifies the published source snapshot against
`SOURCE_MANIFEST.json`; it is an integrity check, not a security guarantee.
