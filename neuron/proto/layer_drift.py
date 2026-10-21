"""Per-layer error growth: CPU bf16 and Neuron bf16, each vs CPU fp32, same Talker inputs.

If CPU bf16 drifts like Neuron, the test inputs are ill-conditioned; if only Neuron drifts,
the compiled graph is wrong. Usage: python neuron/proto/layer_drift.py [--real-kv]
"""
import copy, sys
from pathlib import Path

import torch
import torch.nn as nn
import torch._dynamo

sys.path.insert(0, str(Path(__file__).parent))
from cp_graph import MODEL  # noqa: E402
from talker_decode import attend_dense, rope  # noqa: E402
from nari_qwen3_tts.config import ModelAssetConfig  # noqa: E402
from nari_qwen3_tts.model.checkpoint import CheckpointLoader  # noqa: E402

torch._dynamo.config.automatic_dynamic_shapes = False


class Layers(nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.t = talker
        a = talker.model.layers[0].self_attn
        self.register_buffer("inv_freq", 1.0 / (a.rope_theta ** (torch.arange(0, a.head_dim, 2, dtype=torch.float32) / a.head_dim)))

    def forward(self, x, kv, lengths):
        h, outs = x, []
        for li, layer in enumerate(self.t.model.layers):
            a = layer.self_attn
            res = h
            q, k, v = a.project_qkv(layer.input_layernorm(h))
            q, k = rope(a.q_norm(q), lengths, self.inv_freq), rope(a.k_norm(k), lengths, self.inv_freq)
            h = res + a.o_proj(attend_dense(q, k, v, kv[li, :, 0], kv[li, :, 1], lengths))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
            outs.append(h)
        return torch.stack(outs)


def rel(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-12)).max().item()


def main():
    torch.manual_seed(0)
    loader = CheckpointLoader(ModelAssetConfig(model_id=MODEL, device="cpu", require_h100=False, local_files_only=True))
    talker = loader.load_talker(device=torch.device("cpu"), dtype=torch.bfloat16)
    del talker.model.text_embedding, talker.text_projection
    a = talker.model.layers[0].self_attn
    B, S, L = 32, 512, len(talker.model.layers)
    x = (torch.randn(B, 2048) * 0.05).to(torch.bfloat16)
    kv = (torch.randn(L, B, 2, S, a.num_kv_heads, a.head_dim) * 0.5).to(torch.bfloat16)
    lengths = torch.randint(S // 4, S, (B,))
    with torch.no_grad():
        m = Layers(talker).eval()
        cpu_bf16 = m(x, kv, lengths)
        m32 = copy.deepcopy(m).float()
        ref = m32(x.float(), kv.float(), lengths)
        del m32
        m.to("neuron")
        neu = torch.compile(m, backend="neuron", fullgraph=True)(x.to("neuron"), kv.to("neuron"), lengths.to("neuron")).cpu()
    print("layer | CPU bf16 vs fp32 | Neuron bf16 vs fp32 | Neuron vs CPU bf16 | |h| rms")
    for i in range(L):
        print(f"{i:5d} | {rel(cpu_bf16[i], ref[i]):16.2e} | {rel(neu[i], ref[i]):19.2e} | {rel(neu[i], cpu_bf16[i]):18.2e} | {ref[i].float().pow(2).mean().sqrt().item():8.2f}")


if __name__ == "__main__":
    main()
