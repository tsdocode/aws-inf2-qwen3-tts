"""Prototype: whole-frame Code Predictor (16 depth steps + sampling) as ONE Neuron graph.

Checks (1) greedy tokens match the CPU eager path, (2) per-frame latency at batch B.
Sampling uses host-supplied uniforms (inf2 has no on-device RNG): top-k -> softmax(T) ->
inverse CDF. Usage: python neuron/proto/cp_graph.py --batch 32
"""
import argparse, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from nari_qwen3_tts.config import ModelAssetConfig
from nari_qwen3_tts.model.checkpoint import CheckpointLoader

MODEL = str(Path.home() / "models/Qwen3-TTS-12Hz-1.7B-CustomVoice")


def sample_topk(logits, uniform, temperature: float, top_k: int, greedy: bool):
    if greedy:
        return logits.argmax(-1)
    values, indices = torch.topk(logits.float(), top_k, dim=-1)
    probs = torch.softmax(values / temperature, dim=-1)
    cdf = torch.cumsum(probs, dim=-1)
    pick = (cdf < uniform.unsqueeze(-1) * cdf[:, -1:]).sum(-1).clamp(max=top_k - 1)
    return indices.gather(-1, pick.unsqueeze(-1)).squeeze(-1)


class WholeFrameCodePredictor(nn.Module):
    """Mirrors executor/code_predictor.py:whole_frame with static shapes and no host sync."""

    def __init__(self, predictor, layer0, greedy: bool, temperature=0.9, top_k=50):
        super().__init__()
        self.predictor = predictor
        self.greedy, self.temperature, self.top_k = greedy, temperature, top_k
        with torch.no_grad():
            proj = predictor.small_to_mtp_projection
            residual = predictor.model.codec_embedding
            self.register_buffer("layer0_embed", layer0.weight.detach().clone())
            self.register_buffer("proj_layer0", proj(layer0.weight).detach().contiguous())
            self.register_buffer("proj_residual", torch.stack([proj(residual[i].weight) for i in range(len(residual) - 1)]).detach())
            self.register_buffer("residual_embed", torch.stack([residual[i].weight for i in range(len(residual))]).detach())
        cp = predictor.model.layers[0].self_attn
        self.kv_shape = (len(predictor.model.layers), 2, 16, cp.num_kv_heads, cp.head_dim)

    def forward(self, past_hidden, layer0_token, uniforms):
        rows = layer0_token.shape[0]
        groups = 16
        kv = torch.zeros((self.kv_shape[0], rows) + self.kv_shape[1:], dtype=past_hidden.dtype, device=past_hidden.device)
        positions = torch.arange(groups, device=past_hidden.device).unsqueeze(0).expand(rows, -1)
        self.predictor.forward_depth_unrolled(past_hidden.unsqueeze(1), positions[:, 0:1], kv, cache_pos=0, apply_final_norm=False)
        codec_sum = F.embedding(layer0_token, self.layer0_embed)
        embed = F.embedding(layer0_token, self.proj_layer0)
        tokens = [layer0_token]
        for group in range(1, groups):
            hidden = self.predictor.forward_depth_unrolled(
                embed.unsqueeze(1), positions[:, group:group + 1], kv, cache_pos=group, inputs_are_projected=True
            ).squeeze(1)
            logits = hidden @ self.predictor.lm_head_weight[group - 1].t()
            token = sample_topk(logits, uniforms[:, group - 1], self.temperature, self.top_k, self.greedy)
            tokens.append(token)
            codec_sum = codec_sum + F.embedding(token, self.residual_embed[group - 1])
            if group < groups - 1:
                embed = F.embedding(token, self.proj_residual[group - 1])
        return torch.stack(tokens, dim=1), codec_sum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--sample", action="store_true", help="time the sampling variant instead of greedy")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    torch.manual_seed(0)
    loader = CheckpointLoader(ModelAssetConfig(model_id=MODEL, device="cpu", require_h100=False, local_files_only=True))
    predictor, layer0 = loader.load_code_predictor(device=torch.device("cpu"), dtype=torch.bfloat16)
    module = WholeFrameCodePredictor(predictor, layer0, greedy=not args.sample).eval()

    B = args.batch
    past_hidden = torch.randn(B, 2048, dtype=torch.bfloat16)
    layer0_token = torch.randint(0, 2048, (B,))
    uniforms = torch.rand(B, 15)

    with torch.no_grad():
        ref_tokens, ref_sum = module(past_hidden, layer0_token, uniforms)
        dev = "neuron"
        module.to(dev)
        compiled = torch.compile(module, backend="neuron", fullgraph=True)
        inputs = (past_hidden.to(dev), layer0_token.to(dev), uniforms.to(dev))
        t0 = time.perf_counter()
        tokens, csum = compiled(*inputs)
        tokens = tokens.cpu()
        print(f"compile+first run: {time.perf_counter() - t0:.1f} s")
        if not args.sample:
            match = (tokens == ref_tokens).float().mean().item()
            print(f"greedy token match vs CPU eager: {match * 100:.2f}%  (per group: "
                  f"{[round((tokens[:, g] == ref_tokens[:, g]).float().mean().item(), 2) for g in range(16)]})")
            print(f"codec_sum max abs diff: {(csum.cpu().float() - ref_sum.float()).abs().max().item():.4f}")
        for _ in range(3):
            compiled(*inputs)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            out = compiled(*inputs)
        out[0].cpu()
        ms = (time.perf_counter() - t0) / args.iters * 1e3
        print(f"B={B} {'sample' if args.sample else 'greedy'}: {ms:.2f} ms / frame-batch  "
              f"(real-time budget 80 ms; {ms / B * 1000:.0f} us per stream-frame)")


if __name__ == "__main__":
    main()
