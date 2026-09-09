import torch
@torch.jit.script
def f(x):
    return x / x.norm(p=2, dim=-1).clamp(min=1e-9).unsqueeze(-1)
print("JIT OK:", float(f(torch.ones(1, 3, device="cuda")).sum()))
