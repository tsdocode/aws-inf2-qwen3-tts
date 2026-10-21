"""Streaming (stateful) Qwen3-TTS codec decoder on inf2: chunk-by-chunk, exact continuation of the stream.

front  (torch.compile, one graph per chunk size F): explicit state tensors in -> out
    quantizer.decode -> pre_conv (hist) -> 8-layer sliding-window transformer (71-slot KV per layer, frame
    position, valid length; math mirrors nari model/incremental_codec.py) -> 2x [transconv (kernel == stride:
    stateless), ConvNeXt (dwconv hist 6)] -> decoder[0] conv (hist 6)
blocks (NKI codec_kernels with ``hist``): 4x [snake_transconv (prev input step), 3x residual_unit (6*d)]
tail   (NKI snake_conv_out with ``hist`` 6)

All state is per row ([B, ...]) so B streams decode in lockstep; rows are independent.
"""

import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F

from codec_kernels import (residual_unit, residual_unit_params, snake_conv_out, snake_conv_out_params,
                           snake_transconv, snake_transconv_params)

torch._dynamo.config.automatic_dynamic_shapes = False


def _rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def _causal_conv_step(module, x, hist):
    """Stride-1 CausalConvNet on fresh positions with explicit left history -> (y, new_hist).

    Written as ONE matmul over all rows (im2col, M = B*T): the compiler lowers a batched Conv1d per batch row and
    re-reads the weights for every row (front was 4 GB of HBM reads / 27 ms per call at B=32).
    """
    conv = module.conv
    pad = int(module.padding)
    T = x.shape[-1]
    comb = torch.cat((hist, x), dim=-1) if pad else x
    k, dil = conv.kernel_size[0], conv.dilation[0]
    if conv.groups == conv.in_channels and conv.groups > 1:                 # depthwise: sum of shifted products
        w = conv.weight[:, 0, :]                                             # [C, k]
        y = sum(comb[:, :, j * dil:j * dil + T] * w[:, j].view(1, -1, 1) for j in range(k))
        y = y + conv.bias.view(1, -1, 1)
    else:
        cols = torch.stack([comb[:, :, j * dil:j * dil + T] for j in range(k)], dim=-1)   # [B, Cin, T, k]
        B, Cin = cols.shape[:2]
        cols = cols.permute(0, 2, 1, 3).reshape(B * T, Cin * k)
        y = cols @ conv.weight.reshape(conv.out_channels, Cin * k).t() + conv.bias
        y = y.reshape(B, T, -1).permute(0, 2, 1)
    return y, (comb[..., comb.shape[-1] - pad:] if pad else hist)


def _transconv_kernel_eq_stride(module, x):
    """CausalTransConvNet with kernel == stride (no overlap) as one matmul over all rows."""
    conv = module.conv
    B, Cin, T = x.shape
    r = conv.stride[0]
    y = x.permute(0, 2, 1).reshape(B * T, Cin) @ conv.weight.reshape(Cin, -1)          # [B*T, Cout*r]
    y = y.reshape(B, T, conv.out_channels, r).permute(0, 2, 1, 3).reshape(B, conv.out_channels, T * r)
    return y + conv.bias.view(1, -1, 1)


class StreamingFront(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.d = decoder
        tf = decoder.pre_transformer
        self.window = int(tf.config.sliding_window)
        self.retained = max(0, self.window - 1)

    def state_shapes(self, B):
        d, tf = self.d, self.d.pre_transformer
        a0 = tf.layers[0].self_attn
        heads = tf.config.num_key_value_heads
        latent = d.upsample[0][0].conv.in_channels
        shapes = {
            "pos": ((B,), torch.long), "ctx": ((B,), torch.long),
            "pre": ((B, d.pre_conv.conv.in_channels, int(d.pre_conv.padding)), torch.float32),
            "tf_k": ((len(tf.layers), B, heads, self.retained, a0.head_dim), torch.float32),
            "tf_v": ((len(tf.layers), B, heads, self.retained, a0.head_dim), torch.float32),
            "dec0": ((B, d.decoder[0].conv.in_channels, int(d.decoder[0].padding)), torch.float32),
        }
        for i, blocks in enumerate(d.upsample):
            shapes[f"cnx{i}"] = ((B, latent, int(blocks[1].dwconv.padding)), torch.float32)
            assert blocks[0].right_pad == 0, "upsample transconv expected stateless (kernel == stride)"
        return shapes

    def forward(self, codes, pos, ctx, pre, tf_k, tf_v, cnx0, cnx1, dec0):
        d = self.d
        tf = d.pre_transformer
        hidden = d.quantizer.decode(codes)
        hidden, pre = _causal_conv_step(d.pre_conv, hidden, pre)
        B, Fr = hidden.shape[0], hidden.shape[2]
        # All transformer linears run on 2-D [B*F, C] activations: with 3-D inputs the compiler emits a batched
        # matmul that reloads the weights for every row (4 GB of HBM reads per call at B=32, F=1).
        hidden = tf.input_proj(hidden.transpose(1, 2).reshape(B * Fr, -1))                    # [B*F, D]
        position_ids = pos.unsqueeze(1) + torch.arange(Fr, device=hidden.device).unsqueeze(0)
        cos, sin = tf.rotary_emb(hidden.view(B, Fr, -1), position_ids)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        R = self.retained
        slots = torch.arange(R, device=hidden.device)
        valid_prior = slots.unsqueeze(0) >= R - ctx.unsqueeze(1)
        prior_pos = position_ids[:, :1] + slots.unsqueeze(0) - R
        key_pos = torch.cat((prior_pos, position_ids), 1)
        valid = torch.cat((valid_prior, torch.ones(B, Fr, dtype=torch.bool, device=hidden.device)), 1)
        qp = position_ids.unsqueeze(-1)
        kp = key_pos.unsqueeze(1)
        mask = (valid.unsqueeze(1) & (kp <= qp) & (kp > qp - self.window)).unsqueeze(1)
        new_k, new_v = [], []
        for li, layer in enumerate(tf.layers):
            res = hidden                                                                       # [B*F, D]
            n = layer.input_layernorm(hidden)
            att = layer.self_attn
            shp = (B, Fr, -1, att.head_dim)
            q = att.q_norm(att.q_proj(n).view(shp)).transpose(1, 2)
            k = att.k_norm(att.k_proj(n).view(shp)).transpose(1, 2)
            v = att.v_proj(n).view(shp).transpose(1, 2)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            all_k = torch.cat((tf_k[li], k), 2)
            all_v = torch.cat((tf_v[li], v), 2)
            new_k.append(all_k[:, :, all_k.shape[2] - R:])
            new_v.append(all_v[:, :, all_v.shape[2] - R:])
            if att.num_key_value_groups > 1:
                all_k = all_k.repeat_interleave(att.num_key_value_groups, 1)
                all_v = all_v.repeat_interleave(att.num_key_value_groups, 1)
            o = F.scaled_dot_product_attention(q, all_k, all_v, attn_mask=mask, scale=att.scaling)
            o = att.o_proj(o.transpose(1, 2).reshape(B * Fr, -1))
            hidden = res + layer.self_attn_layer_scale(o)
            res = hidden
            hidden = res + layer.mlp_layer_scale(layer.mlp(layer.post_attention_layernorm(hidden)))
        hidden = tf.output_proj(tf.norm(hidden)).view(B, Fr, -1).permute(0, 2, 1)
        cnx = [cnx0, cnx1]
        for i, blocks in enumerate(d.upsample):
            hidden = _transconv_kernel_eq_stride(blocks[0], hidden)   # kernel == stride: no overlap state
            cn = blocks[1]
            r = hidden
            h, cnx[i] = _causal_conv_step(cn.dwconv, hidden, cnx[i])
            Tn = h.shape[-1]
            h = cn.pwconv2(cn.act(cn.pwconv1(cn.norm(h.permute(0, 2, 1).reshape(B * Tn, -1)))))   # 2-D: one GEMM
            hidden = r + (cn.gamma * h).view(B, Tn, -1).permute(0, 2, 1)
        hidden, dec0 = _causal_conv_step(d.decoder[0], hidden, dec0)
        return (hidden.contiguous(), pos + Fr, torch.clamp(ctx + Fr, max=R), pre, torch.stack(new_k), torch.stack(new_v),
                cnx[0], cnx[1], dec0)


class StreamingCodecDecoder:
    """B streams decoded in lockstep, chunk by chunk; ``reset_rows`` starts new streams in given rows."""

    def __init__(self, decoder, batch, device="neuron", front_dtype=torch.float32):
        decoder = decoder.float().eval()
        self.front_dtype = front_dtype
        self.B, self.device = batch, device
        self.front_mod = StreamingFront(decoder)
        self.blocks = []
        for i in range(4):
            blk = decoder.decoder[1 + i].block
            tc = snake_transconv_params(blk[0], blk[1])
            tc_static = dict(rate=tc.pop("rate"), split_snake=tc.pop("split_snake"))
            cin = blk[1].conv.in_channels
            units = []
            for unit in blk[2:5]:
                p = residual_unit_params(unit)
                static = dict(dilation=p.pop("dilation"), split_snake=p.pop("split_snake"))
                units.append(({k: v.to(device) for k, v in p.items()}, static, unit.conv1.conv.in_channels))
            self.blocks.append((({k: v.to(device) for k, v in tc.items()}, tc_static, cin), units))
        self.tail_params = {k: v.to(device) for k, v in snake_conv_out_params(decoder.decoder[5], decoder.decoder[6]).items()}
        self.tail_c = decoder.decoder[6].conv.in_channels
        decoder.to(device)
        if front_dtype != torch.float32:
            mod = self.front_mod

            def run(*a):
                with torch.autocast("neuron", dtype=front_dtype):
                    out = mod(*a)
                return tuple(o.float() if o.is_floating_point() else o for o in out)
            self.front = torch.compile(run, backend="neuron", fullgraph=True)
        else:
            self.front = torch.compile(self.front_mod, backend="neuron", fullgraph=True)
        self.state = self._fresh_state()

    def _fresh_state(self):
        st = {k: torch.zeros(shape, dtype=dt).to(self.device) for k, (shape, dt) in self.front_mod.state_shapes(self.B).items()}
        for i, ((_, _, cin), units) in enumerate(self.blocks):
            st[f"tc{i}"] = torch.zeros(self.B, cin, 1).to(self.device)
            for j, (_, static, c) in enumerate(units):
                st[f"ru{i}{j}"] = torch.zeros(self.B, c, 6 * static["dilation"]).to(self.device)
        st["tail"] = torch.zeros(self.B, self.tail_c, 6).to(self.device)
        return st

    @torch.no_grad()
    def __call__(self, codes):
        """codes [B, 16, F] (device) -> wav [B, 1, F*1920]; advances every row's stream state."""
        st = self.state
        out = self.front(codes, st["pos"], st["ctx"], st["pre"], st["tf_k"], st["tf_v"], st["cnx0"], st["cnx1"], st["dec0"])
        hidden = out[0]
        st["pos"], st["ctx"], st["pre"], st["tf_k"], st["tf_v"], st["cnx0"], st["cnx1"], st["dec0"] = out[1:]
        for i, ((tc, tc_static, _), units) in enumerate(self.blocks):
            hidden, st[f"tc{i}"] = snake_transconv(hidden, **tc, **tc_static, hist=st[f"tc{i}"])
            for j, (prm, static, _) in enumerate(units):
                hidden, st[f"ru{i}{j}"] = residual_unit(hidden, **prm, **static, hist=st[f"ru{i}{j}"])
        wav, st["tail"] = snake_conv_out(hidden, **self.tail_params, hist=st["tail"])
        return wav
