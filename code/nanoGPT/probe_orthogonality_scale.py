"""Check the raw magnitude of place_orthogonality_loss() against a
checkpoint before picking a weight."""
import sys

import torch

from model import GPTConfig, AdditiveComposedGPT

out_dir = sys.argv[1] if len(sys.argv) > 1 else 'out-addition-additive-v3'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ckpt = torch.load(f'{out_dir}/ckpt.pt', map_location=device)
gptconf = GPTConfig(**ckpt['model_args'])
gptconf.place_orthogonality_max_place = 15
model = AdditiveComposedGPT(gptconf)
sd = ckpt['model']
for k in list(sd.keys()):
    if k.startswith('_orig_mod.'):
        sd[k[len('_orig_mod.'):]] = sd.pop(k)
model.load_state_dict(sd)
model.eval().to(device)

with torch.no_grad():
    loss = model.place_orthogonality_loss()
    print(f"place_orthogonality_loss (mean squared cross-place cosine sim, 0..14) = {loss.item():.5f}")
