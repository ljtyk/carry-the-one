"""
Stage 2 summary: accuracy vs. training progress for the loop-index-embedding
looped model, compared against the Stage 1 fixed-depth baseline and the
single-point naive-looping (no index embedding) result.

Reuses eval_addition.py's model/meta loading and greedy-decode eval logic.

Usage:
    python eval_stage2_curve.py --out_dir=out-addition-loop-idx --device=cuda
"""
import os
import re
import json

import torch

import eval_addition as ea

out_dir = ea.out_dir
dataset_dir = ea.dataset_dir
plot_path = os.path.join(out_dir, 'stage2_curve.png')
results_path = os.path.join(out_dir, 'stage2_curve_results.json')
baseline_results_path = os.path.join('out-addition', 'eval_results.json')
naive_loop_results_path = os.path.join('out-addition-loop', 'eval_results.json')


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

    indist_examples = ea.load_examples(os.path.join(dataset_dir, 'test_indist.txt'))
    ood_examples = ea.load_examples(os.path.join(dataset_dir, 'test_ood.txt'))

    curve = []
    for iter_num, path in snapshots:
        ckpt = torch.load(path, map_location=ea.device)
        state_dict = ckpt['model']
        for k in list(state_dict.keys()):
            if k.startswith('_orig_mod.'):
                state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
        ea.model.load_state_dict(state_dict)
        ea.model.eval()

        indist_results = ea.eval_examples(indist_examples)
        ood_results = ea.eval_examples(ood_examples)
        indist_acc = sum(r['correct'] for r in indist_results) / len(indist_results)
        ood_acc = sum(r['correct'] for r in ood_results) / len(ood_results)
        high_carry = [r for r in indist_results if r['carries'] >= 9]
        high_carry_acc = sum(r['correct'] for r in high_carry) / len(high_carry) if high_carry else None

        print(f"iter {iter_num:6d}  indist_acc={indist_acc:.4f}  ood_acc={ood_acc:.4f}  high_carry_acc={high_carry_acc}")
        curve.append({'iter': iter_num, 'indist_accuracy': indist_acc, 'ood_accuracy': ood_acc,
                       'high_carry_accuracy': high_carry_acc})

    with open(results_path, 'w') as f:
        json.dump(curve, f, indent=2)
    print(f"saved results to {results_path}")

    import matplotlib.pyplot as plt
    iters = [c['iter'] for c in curve]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(iters, [c['indist_accuracy'] for c in curve], marker='o', label='looped+idx: in-distribution (1-10 digits)')
    ax.plot(iters, [c['ood_accuracy'] for c in curve], marker='s', label='looped+idx: OOD (11-15 digits)')
    ax.plot(iters, [c['high_carry_accuracy'] for c in curve], marker='^', label='looped+idx: in-dist, carries>=9')

    if os.path.exists(baseline_results_path):
        with open(baseline_results_path) as f:
            baseline = json.load(f)
        ax.axhline(baseline['test_indist']['overall_accuracy'], color='gray', linestyle=':', linewidth=1.3,
                    label=f"Stage 1 fixed-depth baseline, in-dist ({baseline['test_indist']['overall_accuracy']:.2f}, at 5k iters)")
        ax.axhline(baseline['test_ood']['overall_accuracy'], color='gray', linestyle='--', linewidth=1.3,
                    label=f"Stage 1 fixed-depth baseline, OOD ({baseline['test_ood']['overall_accuracy']:.2f})")

    if os.path.exists(naive_loop_results_path):
        with open(naive_loop_results_path) as f:
            naive = json.load(f)
        ax.scatter([5000], [naive['test_indist']['overall_accuracy']], color='black', marker='x', s=80, zorder=5,
                   label=f"naive looping (no idx embed), in-dist ({naive['test_indist']['overall_accuracy']:.3f}, at 5k iters)")

    ax.set_xlabel('training iteration')
    ax.set_ylabel('exact-match accuracy')
    ax.set_title('Stage 2: single shared block + loop-index embedding vs. Stage 1 fixed-depth baseline')
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"saved plot to {plot_path}")


if __name__ == '__main__':
    main()
