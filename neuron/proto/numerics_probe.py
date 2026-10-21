"""Localize Neuron-vs-CPU numeric divergence: per-component checks on one Talker layer."""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch._dynamo

# Neuron needs static shapes: a recompile for a new shape must not flip dims to dynamic.
torch._dynamo.config.automatic_dynamic_shapes = False

sys.path.insert(0, str(Path(__file__).parent))
from cp_graph import MODEL  # noqa: E402
from talker_decode import attend_dense, rope  # noqa: E402
from nari_qwen3_tts.config import ModelAssetConfig  # noqa: E402
from nari_qwen3_tts.model.checkpoint import CheckpointLoader  # noqa: E402


class Fn(nn.Module):
    def __init__(self, fn, mods=None):
        super().__init__()
        self.fn, self.mods = fn, nn.ModuleList(mods or [])

    def forward(self, *a):
        return self.fn(*a)


def rel(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def check(name, fn, mods, *inputs):
    torch._dynamo.reset()
    with torch.no_grad():
        ref = fn(*inputs)
        m = Fn(fn, mods).to("neuron")
        out = torch.compile(m, backend="neuron", fullgraph=True)(*[t.to("neuron") for t in inputs])
        print(f"{name:34s} rel L2 err = {rel(out, ref):.2e}", flush=True)
        m.to("cpu")


def main():
    torch.manual_seed(0)
    loader = CheckpointLoader(ModelAssetConfig(model_id=MODEL, device="cpu", require_h100=False, local_files_only=True))
    talker = loader.load_talker(device=torch.device("cpu"), dtype=torch.bfloat16)
    layer = talker.model.layers[3]
    a = layer.self_attn
    inv_freq = 1.0 / (a.rope_theta ** (torch.arange(0, a.head_dim, 2, dtype=torch.float32) / a.head_dim))
    B, S = 32, 512
    x = (torch.randn(B, 2048) * 0.05).to(torch.bfloat16)
    lengths = torch.randint(S // 4, S, (B,))
    kc = (torch.randn(B, S, 8, 128) * 0.5).to(torch.bfloat16)
    vc = (torch.randn(B, S, 8, 128) * 0.5).to(torch.bfloat16)
    q = torch.randn(B, 16, 128).to(torch.bfloat16)
    k = torch.randn(B, 8, 128).to(torch.bfloat16)
    v = torch.randn(B, 8, 128).to(torch.bfloat16)

    check("rmsnorm (input_layernorm)", lambda h: layer.input_layernorm(h), [layer.input_layernorm], x)
    check("linear qkv", lambda h: a.qkv_proj(h), [a.qkv_proj], x)
    check("q_norm (per-head rmsnorm)", lambda t: a.q_norm(t), [a.q_norm], q)
    check("rope", lambda t, p: rope(t, p, inv_freq.to(t.device)), [], q, lengths)
    check("attend_dense (masked softmax)", attend_dense, [], q, k, v, kc, vc, lengths)
    check("mlp (gate/up silu down)", lambda h: layer.mlp(h), [layer.mlp], x)
    check("o_proj", lambda t: a.o_proj(t), [a.o_proj], q.reshape(B, -1))


if __name__ == "__main__":
    main()
