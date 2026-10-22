"""NKI codec front transformer vs the exact streaming CPU reference (StreamingFront's transformer math).

Two consecutive chunks (F=1 then F=4) on B rows with different prior context lengths; real inputs (golden codes
through quantizer -> pre_conv -> input_proj). Usage: NEURON_RT_VISIBLE_CORES=1 python neuron/tests/test_codec_front.py
"""
import argparse, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[1] / "kernels"))
sys.path.insert(0, str(Path(__file__).parents[1] / "proto"))
from codec_front_kernels import front_attention_mask, front_rope_tables, front_transformer, front_transformer_params  # noqa: E402
from codec_stream import _rotate_half  # noqa: E402
from cp_graph import MODEL  # noqa: E402
from nari_qwen3_tts.config import ModelAssetConfig  # noqa: E402
from nari_qwen3_tts.model.checkpoint import CheckpointLoader  # noqa: E402

R = 71


def ref_step(tf, hidden, pos, ctx, tf_k, tf_v):
    """CPU reference: hidden [B, F, 512] (after input_proj) -> (norm(hidden) [B, F, 512], new tf_k, new tf_v)."""
    B, Fr = hidden.shape[:2]
    position_ids = pos.unsqueeze(1) + torch.arange(Fr).unsqueeze(0)
    cos, sin = tf.rotary_emb(hidden, position_ids)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    slots = torch.arange(R)
    valid_prior = slots.unsqueeze(0) >= R - ctx.unsqueeze(1)
    prior_pos = position_ids[:, :1] + slots.unsqueeze(0) - R
    key_pos = torch.cat((prior_pos, position_ids), 1)
    valid = torch.cat((valid_prior, torch.ones(B, Fr, dtype=torch.bool)), 1)
    qp, kp = position_ids.unsqueeze(-1), key_pos.unsqueeze(1)
    mask = (valid.unsqueeze(1) & (kp <= qp) & (kp > qp - 72)).unsqueeze(1)
    nk, nv = [], []
    for li, layer in enumerate(tf.layers):
        res = hidden
        n = layer.input_layernorm(hidden)
        att = layer.self_attn
        shp = (B, Fr, -1, att.head_dim)
        q = att.q_norm(att.q_proj(n).view(shp)).transpose(1, 2)
        k = att.k_norm(att.k_proj(n).view(shp)).transpose(1, 2)
        v = att.v_proj(n).view(shp).transpose(1, 2)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        ak, av = torch.cat((tf_k[li], k), 2), torch.cat((tf_v[li], v), 2)
        nk.append(ak[:, :, -R:]); nv.append(av[:, :, -R:])
        o = F.scaled_dot_product_attention(q, ak, av, attn_mask=mask, scale=att.scaling)
        hidden = res + layer.self_attn_layer_scale(att.o_proj(o.transpose(1, 2).reshape(B, Fr, -1)))
        hidden = hidden + layer.mlp_layer_scale(layer.mlp(layer.post_attention_layernorm(hidden)))
    return tf.norm(hidden), torch.stack(nk), torch.stack(nv)


def to_kernel_state(t):   # [L, B, 16, R, 64] -> [L, 8, 128, B, R] (pair dims: even head 0..63, odd head 64..127)
    L, B = t.shape[:2]
    return t.reshape(L, B, 8, 2, R, 64).permute(0, 2, 3, 5, 1, 4).reshape(L, 8, 128, B, R).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8)
    a = ap.parse_args()
    torch.manual_seed(0)
    dec = CheckpointLoader(ModelAssetConfig(model_id=MODEL, device="cpu", require_h100=False, local_files_only=True)) \
        .load_codec(device=torch.device("cpu"), dtype=torch.float32).float().decoder.eval()
    tf = dec.pre_transformer
    codes = torch.load(Path(__file__).resolve().parents[1] / "golden" / "medium.codes.pt")
    B = a.rows
    prm = front_transformer_params(tf)
    eps, qkn = prm.pop("eps"), prm.pop("qk_norm")
    dev = {k: v.to("neuron") for k, v in prm.items()}
    L = len(tf.layers)
    tf_k = torch.zeros(L, B, 16, R, 64); tf_v = torch.zeros_like(tf_k)
    pos = torch.zeros(B, dtype=torch.long); ctx = torch.zeros(B, dtype=torch.long)
    ks, vs = to_kernel_state(tf_k).to(torch.bfloat16).to("neuron"), to_kernel_state(tf_v).to(torch.bfloat16).to("neuron")
    # give rows different histories: row b first consumes 3*b frames on the CPU reference + kernel alike
    f0 = 0
    for Fr in (1, 4, 4):
        with torch.no_grad():
            cs = torch.stack([codes[(f0 + 7 * b) % 60:(f0 + 7 * b) % 60 + Fr] for b in range(B)]).transpose(1, 2)
            hid = dec.quantizer.decode(cs)
            hid = dec.pre_conv.conv(F.pad(hid, (2, 0)))          # stateless here: same input on both sides
            hid = tf.input_proj(hid.transpose(1, 2))             # [B, F, 512]
            ref, nk, nv = ref_step(tf, hid, pos, ctx, tf_k, tf_v)
        cos, sin = front_rope_tables(pos, Fr, tf.rotary_emb.inv_freq)
        mask = front_attention_mask(ctx, Fr)
        t0 = time.perf_counter()
        out, ks, vs = front_transformer(hid.reshape(B * Fr, 512).t().contiguous().to("neuron"), cos.to("neuron"),
                                        sin.to("neuron"), mask.to("neuron"), ks, vs, **dev, eps=eps, qk_norm=qkn)
        got = out.cpu().t().reshape(B, Fr, 512)
        dt = time.perf_counter() - t0
        rel = ((got - ref).norm() / ref.norm()).item()
        srel = ((ks.cpu().float() - to_kernel_state(nk)).norm() / to_kernel_state(nk).norm().clamp_min(1e-9)).item()
        print(f"chunk F={Fr}: hidden rel L2 {rel:.2e} | new K state rel L2 {srel:.2e} | {dt * 1e3:.0f} ms incl compile", flush=True)
        tf_k, tf_v = nk, nv
        pos, ctx = pos + Fr, torch.clamp(ctx + Fr, max=R)
        f0 += Fr


if __name__ == "__main__":
    main()
