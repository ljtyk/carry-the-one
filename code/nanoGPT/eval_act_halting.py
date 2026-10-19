"""
Stage 2.1 characterization: does ACT's per-token halting actually scale with
problem difficulty? For each test example, teacher-force the full correct
sequence ("\na+b=c\n") through the model, read off the per-position ponder
step count at the answer-digit positions, and average them into one
"halting step" value per problem -- then compare against that problem's
carry count and digit count.

Teacher-forcing (not greedy generation) so halting-step is measured against
the true problem structure regardless of whether the model gets it right,
decoupling "how much compute did it use" from "was it correct."

Usage:
    python eval_act_halting.py --out_dir=out-addition-act --device=cuda
"""
import os
import json

import torch

import eval_addition as ea

out_dir = ea.out_dir
dataset_dir = ea.dataset_dir
plot_path = os.path.join(out_dir, 'act_halting_vs_difficulty.png')
results_path = os.path.join(out_dir, 'act_halting_results.json')


@torch.no_grad()
def measure_halting(examples):
    results = []
    for ex in examples:
        prompt = f"\n{ex['a']}+{ex['b']}="
        answer = str(ex['c'])
        full = prompt + answer + '\n'
        ids = ea.encode(full)
        x = torch.tensor([ids], dtype=torch.long, device=ea.device)
        ea.model(x)  # populates ea.model._last_n_updates, shape (1, seq_len)

        n_updates = ea.model._last_n_updates[0]  # (seq_len,)
        answer_start = len(ea.encode(prompt))
        answer_end = answer_start + len(answer)  # exclude trailing newline
        halting_steps = n_updates[answer_start:answer_end]

        results.append({
            **ex,
            'mean_halting_step': halting_steps.mean().item(),
            'max_halting_step': halting_steps.max().item(),
        })
    return results


def summarize_by(results, key):
    buckets = {}
    for r in results:
        buckets.setdefault(r[key], []).append(r['mean_halting_step'])
    return {k: sum(v) / len(v) for k, v in sorted(buckets.items())}


def main():
    indist_examples = ea.load_examples(os.path.join(dataset_dir, 'test_indist.txt'))
    ood_examples = ea.load_examples(os.path.join(dataset_dir, 'test_ood.txt'))

    print(f"measuring halting steps on {len(indist_examples)} in-dist examples...")
    indist_results = measure_halting(indist_examples)
    print(f"measuring halting steps on {len(ood_examples)} OOD examples...")
    ood_results = measure_halting(ood_examples)

    out = {}
    for name, results in [('test_indist', indist_results), ('test_ood', ood_results)]:
        by_carries = summarize_by(results, 'carries')
        by_digits = summarize_by(results, 'digits')
        overall = sum(r['mean_halting_step'] for r in results) / len(results)
        print(f"\n=== {name} ===")
        print(f"overall mean halting step: {overall:.3f}")
        print(f"by carries: {by_carries}")
        print(f"by digits: {by_digits}")
        out[name] = {'overall_mean_halting_step': overall, 'by_carries': by_carries, 'by_digits': by_digits}

    with open(results_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved results to {results_path}")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    for name, marker in [('test_indist', 'o'), ('test_ood', 's')]:
        d = out[name]['by_digits']
        ax.plot(list(d.keys()), list(d.values()), marker=marker, label=name)
    ax.axvline(10.5, color='gray', linestyle='--', linewidth=1, label='train/OOD boundary')
    ax.set_xlabel('digit count')
    ax.set_ylabel('mean halting step (answer-digit positions)')
    ax.set_title('Halting step vs. digit count')
    ax.legend()

    ax = axes[1]
    for name, marker in [('test_indist', 'o'), ('test_ood', 's')]:
        d = out[name]['by_carries']
        ax.plot(list(d.keys()), list(d.values()), marker=marker, label=name)
    ax.set_xlabel('carry count')
    ax.set_ylabel('mean halting step (answer-digit positions)')
    ax.set_title('Halting step vs. carry count')
    ax.legend()

    fig.suptitle('Stage 2.1: does ACT allocate more compute to harder problems?')
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f"saved plot to {plot_path}")


if __name__ == '__main__':
    main()
