"""
Diagnostic: does the fixed-depth baseline's accuracy collapse on carry-heavy
examples even when digit count stays within the trained range (1-10)?

If accuracy vs. carry-count looks about the same on natural-random carries
(test_indist, from eval_addition.py) as on adversarial near-maximal carries
(test_carry_heavy, same digit range) -- controlling for carry count -- that
argues for "learned an actual carry-propagation procedure". If accuracy
craters on the adversarial set even at carry counts/digit lengths where the
natural-random set still does fine, that argues the model memorized common
carry patterns rather than the operation itself.

Reuses eval_addition.py's model loading / greedy decode / example loading.

Usage:
    python eval_carry_heavy.py --out_dir=out-addition --device=cuda
"""
import os
import json

import eval_addition as ea

out_dir = ea.out_dir
dataset_dir = ea.dataset_dir
plot_path = os.path.join(out_dir, 'carry_heavy_comparison.png')
results_path = os.path.join(out_dir, 'carry_heavy_results.json')
indist_results_path = os.path.join(out_dir, 'eval_results.json')


def main():
    path = os.path.join(dataset_dir, 'test_carry_heavy.txt')
    examples = ea.load_examples(path)
    results = ea.eval_examples(examples)
    overall_acc = sum(r['correct'] for r in results) / len(results)
    acc_by_digits = ea.summarize(results, 'digits')
    acc_by_carries = ea.summarize(results, 'carries')

    print(f"\n=== test_carry_heavy ===")
    print(f"overall accuracy: {overall_acc:.4f} (n={len(results)})")
    print(f"accuracy by digit count: {acc_by_digits}")
    print(f"accuracy by carry count: {acc_by_carries}")

    out = {
        'overall_accuracy': overall_acc,
        'accuracy_by_digits': acc_by_digits,
        'accuracy_by_carries': acc_by_carries,
        'n': len(results),
    }
    with open(results_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"saved results to {results_path}")

    if not os.path.exists(indist_results_path):
        print(f"no {indist_results_path} found, skipping comparison plot (run eval_addition.py first)")
        return

    with open(indist_results_path) as f:
        indist = json.load(f)['test_indist']

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    d1 = indist['accuracy_by_digits']
    d2 = acc_by_digits
    ax.plot([int(k) for k in d1.keys()], list(d1.values()), marker='o', label='test_indist (natural carries)')
    ax.plot([int(k) for k in d2.keys()], list(d2.values()), marker='^', label='test_carry_heavy (adversarial)')
    ax.set_xlabel('digit count')
    ax.set_ylabel('exact-match accuracy')
    ax.set_title('Accuracy vs. digit count\n(both within trained 1-10 digit range)')
    ax.set_ylim(-0.02, 1.02)
    ax.legend()

    ax = axes[1]
    c1 = indist['accuracy_by_carries']
    c2 = acc_by_carries
    ax.plot([int(k) for k in c1.keys()], list(c1.values()), marker='o', label='test_indist (natural carries)')
    ax.plot([int(k) for k in c2.keys()], list(c2.values()), marker='^', label='test_carry_heavy (adversarial)')
    ax.set_xlabel('carry count')
    ax.set_ylabel('exact-match accuracy')
    ax.set_title('Accuracy vs. carry count,\ncontrolling for carry count directly')
    ax.set_ylim(-0.02, 1.02)
    ax.legend()

    fig.suptitle('Pattern-matching vs. learned-procedure diagnostic')
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f"saved comparison plot to {plot_path}")


if __name__ == '__main__':
    main()
