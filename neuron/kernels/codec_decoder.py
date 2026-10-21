"""Whole-sequence Qwen3-TTS 12 Hz codec decoder on inf2: compiled frame-rate front + NKI audio-rate blocks.

front  (torch.compile, fp32): quantizer.decode -> pre_conv -> pre_transformer -> 2x(transconv + ConvNeXt)
                               -> decoder[0] conv 1024->1536            (1-4 positions per frame: cheap)
blocks (NKI, codec_kernels):   4x [snake_transconv, 3x residual_unit]   (32..1920 positions per frame)
tail   (NKI):                  SnakeBeta(96) -> conv 96->1 -> clamp   (compiled: 46 ms/chunk, 25%)

Mirrors upstream ``Qwen3TTSTokenizerV2Decoder.forward`` (whole sequence, causal zero history).
"""

import torch
import torch._dynamo
import torch.nn as nn

from codec_kernels import (residual_unit, residual_unit_params, snake_conv_out, snake_conv_out_params,
                           snake_transconv, snake_transconv_params)

torch._dynamo.config.automatic_dynamic_shapes = False


class _Front(nn.Module):
    def __init__(self, decoder, dtype=torch.float32):
        super().__init__()
        self.d, self.dtype = decoder, dtype

    def forward(self, codes):
        d = self.d
        if self.dtype != torch.float32:
            with torch.autocast("neuron", dtype=self.dtype):
                return self._run(d, codes).float()
        return self._run(d, codes)

    @staticmethod
    def _run(d, codes):
        hidden = d.quantizer.decode(codes)
        hidden = d.pre_conv(hidden).transpose(1, 2)
        hidden = d.pre_transformer(inputs_embeds=hidden).last_hidden_state.permute(0, 2, 1)
        for blocks in d.upsample:
            for block in blocks:
                hidden = block(hidden)
        return d.decoder[0](hidden).contiguous()


class _Tail(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.snake, self.conv = decoder.decoder[5], decoder.decoder[6]

    def forward(self, hidden):
        return self.conv(self.snake(hidden)).clamp(min=-1, max=1)


class NeuronCodecDecoder:
    def __init__(self, decoder, device="neuron", front_dtype=torch.float32):
        decoder = decoder.float().eval()
        self.blocks = []
        for i in range(4):
            blk = decoder.decoder[1 + i].block
            tc = snake_transconv_params(blk[0], blk[1])
            tc_static = dict(rate=tc.pop("rate"), split_snake=tc.pop("split_snake"))
            units = []
            for unit in blk[2:5]:
                p = residual_unit_params(unit)
                static = dict(dilation=p.pop("dilation"), split_snake=p.pop("split_snake"))
                units.append(({k: v.to(device) for k, v in p.items()}, static))
            self.blocks.append((({k: v.to(device) for k, v in tc.items()}, tc_static), units))
        tail = snake_conv_out_params(decoder.decoder[5], decoder.decoder[6])
        self.tail_params = {k: v.to(device) for k, v in tail.items()}
        decoder.to(device)
        self.front = torch.compile(_Front(decoder, front_dtype), backend="neuron", fullgraph=True)

    def tail(self, hidden):
        return snake_conv_out(hidden, **self.tail_params)

    @torch.no_grad()
    def __call__(self, codes):
        hidden = self.front(codes)
        for (tc, tc_static), units in self.blocks:
            hidden = snake_transconv(hidden, **tc, **tc_static)
            for prm, static in units:
                hidden = residual_unit(hidden, **prm, **static)
        return self.tail(hidden)


def stage_breakdown(neu, codes, iters=5):
    """Wall ms per stage (each synchronized), for finding the next bottleneck."""
    import time

    def timed(fn, *args):
        out = fn(*args)
        torch.neuron.synchronize()  # device sync without a D2H copy (block3 output alone is ~377 MB)
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fn(*args)
        torch.neuron.synchronize()
        return out, (time.perf_counter() - t0) / iters * 1e3

    rows = []
    with torch.no_grad():
        hidden, ms = timed(neu.front, codes)
        rows.append(("front (compiled)", ms))
        for i, ((tc, tc_static), units) in enumerate(neu.blocks):
            hidden, ms = timed(lambda h: snake_transconv(h, **tc, **tc_static), hidden)
            rows.append((f"block{i} snake+transconv", ms))
            for j, (prm, static) in enumerate(units):
                hidden, ms = timed(lambda h: residual_unit(h, **prm, **static), hidden)
                rows.append((f"block{i} ru{j} (d={static['dilation']})", ms))
        _, ms = timed(neu.tail, hidden)
        rows.append(("tail (compiled)", ms))
    return rows
