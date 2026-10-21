"""Prototype: Talker decode step (28 layers + codec_head + sampling) as one Neuron graph.

Dense per-row KV [L, B, 2, S, KVH, D] read-only inside the graph (lengths masked); the new K/V
is returned so the cache write can be a separate kernel (in-graph indexed writes become
full-buffer scatters on Neuron). Reference = the repo's own Talker layers on CPU through a
dense attention context. Usage: python neuron/proto/talker_decode.py --batch 32 --ctx 512
"""
import argparse, sys, time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from cp_graph import MODEL, sample_topk  # noqa: E402
from nari_qwen3_tts.config import ModelAssetConfig  # noqa: E402
from nari_qwen3_tts.model.checkpoint import CheckpointLoader  # noqa: E402


def rope(x, pos, inv_freq):  # x [B,H,D], pos [B]
    ang = pos.float().unsqueeze(-1) * inv_freq                      # [B, D/2]
    cos = torch.cat((ang.cos(), ang.cos()), -1).unsqueeze(1)
    sin = torch.cat((ang.sin(), ang.sin()), -1).unsqueeze(1)
    a, b = x.float().chunk(2, -1)
    return (x.float() * cos + torch.cat((-b, a), -1) * sin).to(x.dtype)


def attend_dense(q, k_new, v_new, k_cache, v_cache, lengths):
    """q [B,H,D]; k/v_new [B,KVH,D]; caches [B,S,KVH,D]; attends cache[:len] + new token."""
    B, H, D = q.shape
    kvh = k_new.shape[1]
    rep = H // kvh
    qg = q.float().view(B, kvh, rep, D)
    s_cache = torch.einsum("bgrd,bsgd->bgrs", qg, k_cache.float())
    valid = torch.arange(k_cache.shape[1], device=q.device).view(1, 1, 1, -1) < lengths.view(-1, 1, 1, 1)
    s_cache = s_cache.masked_fill(~valid, -torch.inf)
    s_new = torch.einsum("bgrd,bgd->bgr", qg, k_new.float()).unsqueeze(-1)
    probs = torch.softmax(torch.cat((s_cache, s_new), -1) * D ** -0.5, -1)
    out = torch.einsum("bgrs,bsgd->bgrd", probs[..., :-1], v_cache.float()) + probs[..., -1:] * v_new.float().unsqueeze(2)
    return out.reshape(B, H * D).to(q.dtype)


class TalkerDecode(nn.Module):
    def __init__(self, talker, greedy: bool, temperature=0.9, top_k=50):
        super().__init__()
        self.t, self.greedy, self.temperature, self.top_k = talker, greedy, temperature, top_k
        a = talker.model.layers[0].self_attn
        self.register_buffer("inv_freq", 1.0 / (a.rope_theta ** (torch.arange(0, a.head_dim, 2, dtype=torch.float32) / a.head_dim)))

    def forward(self, x, kv, lengths, uniform):
        # x [B,2048] input embedding; kv [L,B,2,S,KVH,D]; lengths [B] (= position of new token)
        h, new_kv = x, []
        for li, layer in enumerate(self.t.model.layers):
            a = layer.self_attn
            res = h
            q, k, v = a.project_qkv(layer.input_layernorm(h))
            q, k = rope(a.q_norm(q), lengths, self.inv_freq), rope(a.k_norm(k), lengths, self.inv_freq)
            new_kv.append(torch.stack((k, v), 1))
            h = res + a.o_proj(attend_dense(q, k, v, kv[li, :, 0], kv[li, :, 1], lengths))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        hidden = self.t.model.norm(h)
        logits = self.t.codec_head(hidden)
        token = sample_topk(logits, uniform, self.temperature, self.top_k, self.greedy)
        return token, hidden, torch.stack(new_kv)


class DenseContext:
    """Minimal TalkerAttentionContext over the same dense cache, driving the repo's own layers."""

    def __init__(self, kv, lengths, inv_freq):
        self.kv, self.lengths, self.inv_freq, self.layer = kv, lengths, inv_freq, 0

    def select_layer(self, i):
        self.layer = i

    def advance_sequence_lengths(self):
        pass

    def attend(self, module, query, key, value):
        q = rope(module.q_norm(query), self.lengths, self.inv_freq)
        k = rope(module.k_norm(key), self.lengths, self.inv_freq)
        return attend_dense(q, k, value, self.kv[self.layer, :, 0], self.kv[self.layer, :, 1], self.lengths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    torch.manual_seed(0)
    loader = CheckpointLoader(ModelAssetConfig(model_id=MODEL, device="cpu", require_h100=False, local_files_only=True))
    talker = loader.load_talker(device=torch.device("cpu"), dtype=torch.bfloat16)
    del talker.model.text_embedding, talker.text_projection  # prefill-only; keep device memory for KV
    m = TalkerDecode(talker, greedy=True).eval()
    a = talker.model.layers[0].self_attn
    B, S, L = args.batch, args.ctx, len(talker.model.layers)
    x = (torch.randn(B, 2048) * 0.05).to(torch.bfloat16)
    kv = (torch.randn(L, B, 2, S, a.num_kv_heads, a.head_dim) * 0.5).to(torch.bfloat16)
    lengths = torch.randint(S // 4, S, (B,))
    uniform = torch.rand(B)

    with torch.no_grad():
        ref_hidden = talker.model(x, DenseContext(kv, lengths, m.inv_freq))       # repo layers, CPU
        mine_tok, mine_hidden, _ = m(x, kv, lengths, uniform)                     # prototype math, CPU
        print(f"CPU prototype vs repo layers: hidden max abs diff {(mine_hidden.float() - ref_hidden.float()).abs().max().item():.4f}")
        ref_logits = talker.codec_head(ref_hidden.to(torch.bfloat16)).float()
        m.to("neuron")
        c = torch.compile(m, backend="neuron", fullgraph=True)
        ins = [t.to("neuron") for t in (x, kv, lengths, uniform)]
        t0 = time.perf_counter()
        tok, hidden, new_kv = c(*ins)
        tok = tok.cpu()
        print(f"compile + first step: {time.perf_counter() - t0:.1f} s")
        top2 = ref_logits.topk(2, -1).values
        margin = (top2[:, 0] - top2[:, 1])
        agree = (tok == ref_logits.argmax(-1))
        print(f"greedy token agreement vs repo CPU: {agree.float().mean().item() * 100:.1f}%  "
              f"(disagreements' top1-top2 logit margin: {margin[~agree].tolist()[:8]})")
        rel = ((hidden.cpu().float() - ref_hidden.float()).norm(dim=-1) / ref_hidden.float().norm(dim=-1)).max().item()
        print(f"final hidden max relative L2 error: {rel:.4f}")
        for _ in range(3):
            c(*ins)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            out = c(*ins)
        out[0].cpu()
        ms = (time.perf_counter() - t0) / args.iters * 1e3
        print(f"B={B} ctx={S}: {ms:.2f} ms per decode step")


if __name__ == "__main__":
    main()
