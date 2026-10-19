"""Walk through the HRR codec mechanism with concrete numbers: a tiny
toy-dimension example you can basically read by eye, then the same
mechanism at full scale (dim=384) using v17's actual trained weights,
showing the crosstalk noise directly and how the correct digit still
wins the readout despite it."""
import math
import sys

import torch
import torch.nn.functional as F

from model import GPTConfig, AdditiveComposedGPT, _precompute_unitary_roles, _circular_bind, _circular_unbind

torch.set_printoptions(precision=3, sci_mode=False)


def toy_example():
    print("=" * 70)
    print("PART 1: toy example, dim=8, so every vector fits on one line")
    print("=" * 70)
    dim = 8
    roles = _precompute_unitary_roles(n_places=4, dim=dim, seed=0)
    print(f"\nRole vectors (fixed, non-learned, one per place, shape ({4},{dim})):")
    for p in range(4):
        print(f"  role[{p}] = {roles[p]}   ||role[{p}]|| = {roles[p].norm():.4f}")

    torch.manual_seed(1)
    digit_vecs = torch.randn(3, dim) * 0.5  # pretend these are digit_embed[4], digit_embed[8], digit_embed[2]
    labels = ['4', '8', '2']
    print(f"\nThree toy 'digit identity' vectors (stand-ins for digit_embed rows):")
    for lbl, v in zip(labels, digit_vecs):
        print(f"  vec('{lbl}') = {v}")

    print("\n--- binding a single digit to a place ---")
    f, r = digit_vecs[0], roles[0]
    bound = _circular_bind(f, r)
    print(f"  bound = circular_bind(vec('4'), role[0]) = {bound}")
    print(f"  ||bound|| = {bound.norm():.4f}  (||vec('4')||={f.norm():.4f} -- binding preserves norm exactly)")

    recovered = _circular_unbind(bound, r)
    print(f"\n--- unbinding recovers it EXACTLY (no other terms yet) ---")
    print(f"  circular_unbind(bound, role[0]) = {recovered}")
    print(f"  max abs error vs original vec('4'): {(recovered - f).abs().max().item():.2e}")

    print("\n--- now superpose a 3-digit number: places 2,1,0 = digits 4,8,2 ---")
    vec = (_circular_bind(digit_vecs[0], roles[2])   # '4' at place 2 (hundreds)
           + _circular_bind(digit_vecs[1], roles[1])  # '8' at place 1 (tens)
           + _circular_bind(digit_vecs[2], roles[0]))  # '2' at place 0 (units)
    print(f"  vec (single 8-dim vector encoding the whole number '482') = {vec}")

    print("\n--- decode each place: unbind, then compare against the true digit vector ---")
    for true_lbl, true_vec, place in zip(labels, digit_vecs, [2, 1, 0]):
        q = _circular_unbind(vec, roles[place])
        crosstalk = q - true_vec
        print(f"\n  place {place} (true digit '{true_lbl}'):")
        print(f"    unbind(vec, role[{place}])      = {q}")
        print(f"    true vec('{true_lbl}')            = {true_vec}")
        print(f"    crosstalk (= difference)     = {crosstalk}")
        print(f"    ||crosstalk|| / ||true_vec|| = {crosstalk.norm() / true_vec.norm():.3f}"
              f"  <- noise from the OTHER 2 superposed terms, at 8 dims this is already big")


@torch.no_grad()
def real_example(out_dir='out-addition-additive-v17'):
    print("\n" + "=" * 70)
    print(f"PART 2: the real thing, dim=384, weights from {out_dir}")
    print("=" * 70)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(f'{out_dir}/ckpt.pt', map_location=device)
    gptconf = GPTConfig(**ckpt['model_args'])
    model = AdditiveComposedGPT(gptconf)
    sd = ckpt['model']
    for k in list(sd.keys()):
        if k.startswith('_orig_mod.'):
            sd[k[len('_orig_mod.'):]] = sd.pop(k)
    model.load_state_dict(sd)
    model.eval().to(device)

    def show_number(s, label):
        print(f"\n--- encoding '{s}' ({label}) ---")
        digit_ids = torch.tensor([[int(c) for c in s]], dtype=torch.long, device=device)
        vec = model.encode_operand(digit_ids)  # (1, 384) -- the WHOLE number, one vector
        print(f"  encoded vector: shape {tuple(vec.shape)}, ||vec|| = {vec.norm().item():.3f}")

        L = len(s)
        places = list(range(L - 1, -1, -1))  # units = place 0
        print(f"\n  decoding each place (true digit -> top-3 candidates by cosine similarity"
              f" against all 11 digit_embed/blank rows):")
        class_vectors = torch.cat([model.digit_embed.weight, model.blank_embed.unsqueeze(0)], dim=0)  # (11, C)
        class_labels = [str(d) for d in range(10)] + ['_']
        n_wrong = 0
        for ch, place in zip(s, places):
            role = model.place_signal[place].unsqueeze(0)
            q = _circular_unbind(vec, role)  # (1, 384) -- this place's recovered content, true signal + crosstalk
            sims = F.cosine_similarity(q, class_vectors.unsqueeze(0), dim=-1).squeeze(0)  # (11,)
            top3 = torch.topk(sims, 3)
            pred = class_labels[top3.indices[0].item()]
            mark = "OK" if pred == ch else "WRONG"
            top3_str = ", ".join(f"'{class_labels[i]}'={v:.3f}" for i, v in zip(top3.indices.tolist(), top3.values.tolist()))
            if pred != ch:
                n_wrong += 1
            print(f"    place {place:2d}  true='{ch}'  top3: [{top3_str}]  [{mark}]")
        print(f"  -> {L - n_wrong}/{L} places correct")

    show_number("482", "short, well within training range")
    show_number("14235897160293847561", "L=20, right at the edge of training range")
    show_number("142358971602938475619384756", "L=27, deep in far-OOD collapse territory")


if __name__ == '__main__':
    toy_example()
    real_example(sys.argv[1] if len(sys.argv) > 1 else 'out-addition-additive-v17')
