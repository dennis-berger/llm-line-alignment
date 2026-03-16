This directory vendors the PyLaia runtime assets needed by the Bullinger NNTP baseline.

Included files:

- `epoch=170-lowest_va_cer.ckpt`
- `model`
- `syms.txt`

Source:

- copied from the sibling directory `../pylaia-dennis` on 2026-03-16

Why only these files:

- `scripts/run_nntp_eval.py` needs the checkpoint, `model`, and `syms.txt`
- the original `netout_edited.yaml` is not required because the runner generates a fresh netout config per run
- the original `read_lattice.py` and `process_lattices.py` are not required because their useful logic has been reimplemented under `src/linealign/nntp/`

SHA-256:

- `epoch=170-lowest_va_cer.ckpt`: `1d43f424a6c65e775c7d71171f154afbbeb690ee3a1ede1fb60a3e9ac736e2cf`
- `model`: `a6d51e203cdd6f3e24b56fc3a0dd226ba700c60d051f76c3883d87244616cf6d`
- `syms.txt`: `58a95c71338a633e818663aa0743e9a45addddad4c325c99667fee8d32551ed5`
