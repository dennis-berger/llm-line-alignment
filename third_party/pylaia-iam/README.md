This directory vendors the PyLaia runtime assets needed by the IAM RWTH NNTP baseline.

Included files:

- `model`
- `syms.txt`
- `weights.ckpt`

Source:

- downloaded from the public Hugging Face model `Teklia/pylaia-iam` on 2026-03-18
- model page: <https://huggingface.co/Teklia/pylaia-iam>

Why only these files:

- `scripts/run_nntp_eval.py` needs the checkpoint, `model`, and `syms.txt`
- `language_model.arpa.gz`, `lexicon.txt`, and `tokens.txt` are optional for PyLaia decoding, but they are not required for the NNTP forced-alignment path

SHA-256:

- `model`: `cfe0c2e1da98dcddb0db65e11635b5fbb6a620d07529c29ef027ea1293aee784`
- `syms.txt`: `79cfd87cabdf7a137b7b3c9814481b17b1ce4cd52717748a9390da61f0e6c55f`
- `weights.ckpt`: `9b9541eb80007bc817bbe5b91828f3dc3ddc7e461d3480bf14cc6931458474b2`
