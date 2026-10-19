"""
Track in-distribution and OOD exact-match accuracy across the numbered
snapshots saved by a snapshot_interval-enabled training run (see
config/train_addition_grok.py), to see whether/when OOD (length-extrapolation)
accuracy moves at all over the course of a long, heavy-weight-decay run.

Also tracks accuracy on the high-carry (carries >= 9) subset of the
in-distribution test set specifically. This is a separate, more sensitive
signal: if weight decay is consolidating memorized carry-shortcuts into
something closer to a genuine length-invariant procedure (the "grokking"
hypothesis), that should show up here first -- within the trained digit
range, where the model isn't also fighting unfamiliar sequence lengths --
even if OOD extrapolation itself never moves.

Reuses eval_addition.py's model/meta loading and greedy-decode eval logic,
swapping in each snapshot's weights in turn.

Usage:
    python eval_grok_curve.py --out_dir=out-addition-grok --device=cuda
"""
import os
import re
import json

import torch

import eval_addition as ea

out_dir = ea.out_dir  # picked up from --out_dir via eval_addition's own configurator run at import time
dataset_dir = ea.dataset_dir
plot_path = os.path.join(out_dir, 'grok_curve.png')
results_path = os.path.join(out_dir, 'grok_curve_results.json')


def find_snapshots(d):
    pattern = re.compile(r'ckpt_(\d+)\.pt$')
    snaps = []
    for fname in os.listdir(d):
        m = pattern.search(fname)
        if m:
            snaps.append((int(m.group(1)), os.path.join(d, fname)))
    return sorted(snaps)


def main():
    snapshots = find_snapshots(out_dir)
    print(f"found {len(snapshots)} snapshots in {out_dir}")
    if not snapshots:
        print("no snapshots found -- did the run use snapshot_interval > 0?")
        return

    indist_examples = ea.load_examples(os.path.join(dataset_dir, 'test_indist.txt'))
    ood_examples = ea.load_examples(os.path.join(dataset_dir, 'test_ood.txt'))

    curve = []
    for iter_num, path in snapshots:
        ckpt = torch.load(path, map_location=ea.device)
        state_dict = ckpt['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        ea.model.load_state_dict(state_dict)
        ea.model.eval()

        indist_results = ea.eval_examples(indist_examples)
        ood_results = ea.eval_examples(ood_examples)
        indist_acc = sum(r['correct'] for r in indist_results) / len(indist_results)
        ood_acc = sum(r['correct'] for r in ood_results) / len(ood_results)

        high_carry = [r for r in indist_results if r['carries'] >= 9]
        high_carry_acc = sum(r['correct'] for r in high_carry) / len(high_carry) if high_carry else None

        print(f"iter {iter_num:7d}  indist_acc={indist_acc:.4f}  ood_acc={ood_acc:.4f}  "
              f"high_carry_acc={high_carry_acc if high_carry_acc is None else round(high_carry_acc, 4)} (n={len(high_carry)})")
        curve.append({
            'iter': iter_num,
            'indist_accuracy': indist_acc,
            'ood_accuracy': ood_acc,
            'high_carry_accuracy': high_carry_acc,
            'high_carry_n': len(high_carry),
        })

    with open(results_path, 'w') as f:
        json.dump(curve, f, indent=2)
    print(f"saved results to {results_path}")

    import matplotlib.pyplot as plt
    iters = [c['iter'] for c in curve]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(iters, [c['indist_accuracy'] for c in curve], marker='o', markersize=3, label='in-distribution (1-10 digits)')
    ax.plot(iters, [c['ood_accuracy'] for c in curve], marker='s', markersize=3, label='OOD (11-15 digits)')
    ax.plot(iters, [c['high_carry_accuracy'] for c in curve], marker='^', markersize=3,
            label='in-distribution, carries>=9')

    baseline_path = os.path.join('out-addition', 'eval_results.json')
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        ax.axhline(baseline['test_indist']['overall_accuracy'], color='gray', linestyle=':', linewidth=1.3,
                    label=f"Stage 1 fixed-depth baseline, in-dist ({baseline['test_indist']['overall_accuracy']:.2f})")
        ax.axhline(baseline['test_ood']['overall_accuracy'], color='gray', linestyle='--', linewidth=1.3,
                    label=f"Stage 1 fixed-depth baseline, OOD ({baseline['test_ood']['overall_accuracy']:.2f})")

    if any(i <= 34000 for i in iters) and any(i > 34000 for i in iters):
        ax.axvline(34000, color='tab:purple', linestyle='--', linewidth=1, alpha=0.6)
        ax.annotate('weight_decay 1.0->0.1', xy=(34000, 1.0), xytext=(2, -10),
                    textcoords='offset points', rotation=90, va='top', fontsize=8, color='tab:purple')
    if any(i <= 58000 for i in iters) and any(i > 58000 for i in iters):
        ax.axvline(58000, color='tab:red', linestyle='--', linewidth=1, alpha=0.6)
        ax.annotate('collapse', xy=(58000, 1.0), xytext=(2, -10),
                    textcoords='offset points', rotation=90, va='top', fontsize=8, color='tab:red')

    ax.set_xlabel('training iteration')
    ax.set_ylabel('exact-match accuracy')
    ax.set_title('Grokking diagnostic: fixed-depth addition model, extended training under weight decay')
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"saved plot to {plot_path}")


if __name__ == '__main__':
    main()
