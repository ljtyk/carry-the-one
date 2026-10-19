"""
Ad-hoc probe: does the additive encoder treat "400" as canonically equal
regardless of how many explicit zero-digit terms are summed in, or does
each additional (even-zero) term meaningfully change vec_a?

Three candidate sums for the "hundreds digit is 4, rest are 0" case:
  v1 = combine_fn(digit=4, place=2)                                   # hundreds only
  v2 = combine_fn(digit=4, place=2) + combine_fn(digit=0, place=1)     # + tens zero
  v3 = combine_fn(digit=4, place=2) + combine_fn(digit=0, place=1) + combine_fn(digit=0, place=0)  # + ones zero (== encode_operand([4,0,0]), i.e. "400")

Reports pairwise cosine/L2 distance and what each decodes to.
"""
import sys
import torch

from model import GPTConfig, AdditiveComposedGPT

out_dir = sys.argv[1] if len(sys.argv) > 1 else 'out-addition-additive-150k'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ckpt = torch.load(f'{out_dir}/ckpt.pt', map_location=device)
gptconf = GPTConfig(**ckpt['model_args'])
model = AdditiveComposedGPT(gptconf)
sd = ckpt['model']
for k in list(sd.keys()):
    if k.startswith('_orig_mod.'):
        sd[k[len('_orig_mod.'):]] = sd.pop(k)

# This checkpoint predates the digit_embed/blank_embed tying change: its
# extract_fn.2 outputs 11 raw logits directly instead of a 384-dim query.
# Pull those raw weights out before loading, and reconstruct the OLD
# decode path manually (combine_fn/encode_operand are unaffected).
old_extract2_w = sd.pop('extract_fn.2.weight').to(device)  # (11, 1536)
old_extract2_b = sd.pop('extract_fn.2.bias').to(device)    # (11,)
model.load_state_dict(sd, strict=False)  # blank_embed stays randomly init, unused below
model.eval().to(device)

import torch.nn.functional as _F


def decode_str(vec):
    """Reconstructs the checkpoint's OLD decode_from_vector: extract_fn's
    first two layers (shared arch) then the OLD final Linear->11 classes,
    instead of the new query/dot-product-against-digit_embed path."""
    B = 1
    P = model.config.max_answer_digits
    places = torch.arange(P - 1, -1, -1, device=device)
    place_sig = model.place_signal[places].unsqueeze(0)
    mem_expanded = vec.view(1, 1, -1).expand(-1, P, -1)
    combined = torch.cat([mem_expanded, place_sig], dim=-1)
    h = _F.gelu(model.extract_fn[0](combined))
    logits = _F.linear(h, old_extract2_w, old_extract2_b)  # (1, P, 11)
    preds = torch.argmax(logits[0], dim=-1).tolist()
    return ''.join(str(p) for p in preds if p != 10)


def term(digit, place):
    d = torch.tensor([[digit]], device=device)
    p = torch.tensor([place], device=device)
    d_emb = model.digit_embed(d)  # (1,1,C)
    place_sig = model.place_signal[p].unsqueeze(0)  # (1,1,C)
    combined = torch.cat([d_emb, place_sig], dim=-1)
    return model.combine_fn(combined).squeeze(0).squeeze(0)  # (C,)


with torch.no_grad():
    t_hundreds = term(4, 2)
    t_tens0 = term(0, 1)
    t_ones0 = term(0, 0)

    v1 = t_hundreds
    v2 = t_hundreds + t_tens0
    v3 = t_hundreds + t_tens0 + t_ones0

    ref = model.encode_operand(torch.tensor([[4, 0, 0]], device=device)).squeeze(0)

    print(f"||t_hundreds(4,place2)|| = {t_hundreds.norm().item():.4f}")
    print(f"||t_tens0(0,place1)||    = {t_tens0.norm().item():.4f}")
    print(f"||t_ones0(0,place0)||    = {t_ones0.norm().item():.4f}")
    print()
    for name, v in [('v1 (hundreds only)', v1), ('v2 (+tens zero)', v2), ('v3 (+ones zero)', v3), ('ref=encode_operand([4,0,0])', ref)]:
        print(f"{name}: ||v||={v.norm().item():.4f}  decodes to '{decode_str(v)}'")
    print()

    def cos(a, b):
        return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    print(f"cos(v1, v2) = {cos(v1, v2):.4f}   L2(v1,v2) = {(v1 - v2).norm().item():.4f}")
    print(f"cos(v2, v3) = {cos(v2, v3):.4f}   L2(v2,v3) = {(v2 - v3).norm().item():.4f}")
    print(f"cos(v1, v3) = {cos(v1, v3):.4f}   L2(v1,v3) = {(v1 - v3).norm().item():.4f}")
    print(f"L2(v3, ref) = {(v3 - ref).norm().item():.8f}  (sanity: should be ~0)")

    # extra: what does a *shorter* number encoded normally look like relative to v1/v2?
    ref40 = model.encode_operand(torch.tensor([[4, 0]], device=device)).squeeze(0)   # "40" as its own natural 2-digit encode
    ref4 = model.encode_operand(torch.tensor([[4]], device=device)).squeeze(0)        # "4" as its own natural 1-digit encode
    print()
    print(f"encode_operand([4])   decodes to '{decode_str(ref4)}'   cos(v1, this)={cos(v1, ref4):.4f}")
    print(f"encode_operand([4,0]) decodes to '{decode_str(ref40)}'  cos(v2, this)={cos(v2, ref40):.4f}")
