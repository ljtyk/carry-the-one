# Carry the One

A research log on length generalization in transformer addition. The post is `index.html`
(open it in a browser); the code behind every result in it is in `code/nanoGPT/`.

- `code/nanoGPT/model.py` — all model variants, including `AdditiveComposedGPT` (encode → combine → decode)
- `code/nanoGPT/train.py` — training; configs for each run are in `code/nanoGPT/config/` (`train_addition_additive_v22.py` is the closed answer, `v35`–`v38` are the single-token experiments)
- `code/nanoGPT/eval_additive.py` — accuracy by operand length: `python eval_additive.py --out_dir=<run dir>`
- `code/nanoGPT/probe_*.py` — the mechanistic probes

Built on [nanoGPT](https://github.com/karpathy/nanoGPT) (MIT, see `code/nanoGPT/LICENSE`).
Trained checkpoints are not included.
