# SPDX-License-Identifier: Apache-2.0
"""Q13: lossless 13-bit resident storage for bf16 linear weights.

Each weight is stored as three fixed-width planes - a sign+mantissa byte, a
4-bit exponent-LUT index nibble, and a 1-bit high-index bitmap - with one
32-entry exponent LUT per 1M-element chunk. Reconstruction is bit-exact bf16,
so this is a quantization method with zero quantization error: ~19% of weight
VRAM back, outputs identical to running unquantized.

Why this works, conceptually: a bf16 weight spends 8 bits on its exponent,
but trained weights live in a narrow band of magnitudes. Take any
million-element chunk and the spread between its biggest and tiniest values
covers a few dozen powers of two at most (measured here: 14-29 distinct
exponents per chunk, never more than 32). Bounding that band per chunk turns
the 8-bit exponent into a 5-bit LUT index. The mantissa is the opposite
case: it is statistically near-random - that is what "precision" is - so it
is kept raw. Sign 1 + mantissa 7 + exponent index 5 = 13 bits.

Why 13 bits and not the ~10.6-bit entropy floor: runtime. An entropy coder
(rANS) reaches the floor but decodes sequentially and gives up random
access; fixed-width planes decode with shifts and masks at memory speed,
inside whatever kernel needs the weight. The last ~2.5 bits are deliberately
left on the table in exchange for a decode that never becomes the
bottleneck. And it is lossless because the contract is precision first:
speed is only taken where it costs nothing.

Measured on two Arc Pro B60s on Windows 11: whole-model packing of Qwen3-1.7B
lands at r = 0.8125 exactly (2051 chunks, zero fallbacks, every tensor
round-tripped bit-exact); the offline smoke is greedy token-identical to bf16,
with model memory 0.93 -> 0.81 GiB on a 0.5B. The fused apply reads the
planes directly inside the matmul (the expanded weight never touches memory);
set VLLM_Q13_FUSED=0 to fall back to unpack + the stock unquantized GEMM.
Enable with quantization="q13" after importing this module; registration is
via the decorator, no core files are touched.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import (
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

logger = init_logger(__name__)

CHUNK = 1 << 20          # elements per exponent-LUT chunk (multiple of 32)
CHUNK_SHIFT = 20
# The fused kernels pass standalone tests (within fp32 reassociation of
# F.linear on real weights, all shapes) but the in-engine wiring is not yet
# validated, so fused apply is OPT-IN experimental: VLLM_Q13_FUSED=1.
# Default path: unpack + the stock unquantized GEMM (token-identical to bf16).
_FUSED = os.environ.get("VLLM_Q13_FUSED", "0") == "1"

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _gemv_q13(sm32_ptr, nib32_ptr, hi32_ptr, lut_ptr, x_ptr, y_ptr,
                  K: tl.constexpr, BK: tl.constexpr,
                  CHUNK_SHIFT: tl.constexpr):
        row = tl.program_id(0)
        acc = tl.zeros((BK,), tl.float32)
        for k0 in range(0, K, BK):
            base = row * K + k0
            i = base + tl.arange(0, BK)
            w4 = tl.load(sm32_ptr + (base >> 2) + tl.arange(0, BK // 4))
            sm = tl.reshape(w4[:, None] >> ((tl.arange(0, 4) * 8)[None, :]), (BK,)) & 0xFF
            w8 = tl.load(nib32_ptr + (base >> 3) + tl.arange(0, BK // 8))
            nb = tl.reshape(w8[:, None] >> ((tl.arange(0, 8) * 4)[None, :]), (BK,)) & 0xF
            w32 = tl.load(hi32_ptr + (base >> 5) + tl.arange(0, BK // 32))
            hb = tl.reshape(w32[:, None] >> (tl.arange(0, 32)[None, :]), (BK,)) & 1
            idx = nb | (hb << 4)
            e = tl.load(lut_ptr + ((i >> CHUNK_SHIFT) << 5) + idx)
            u = ((sm >> 7) << 15) | (e << 7) | (sm & 0x7F)
            wv = u.to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)
            acc += wv * tl.load(x_ptr + k0 + tl.arange(0, BK))
        tl.store(y_ptr + row, tl.sum(acc, axis=0))

    @triton.jit
    def _gemm_q13(sm32_ptr, nib32_ptr, hi32_ptr, lut_ptr, x_ptr, y_ptr,
                  K: tl.constexpr, B: tl.constexpr, BK: tl.constexpr,
                  BN: tl.constexpr, CHUNK_SHIFT: tl.constexpr):
        row = tl.program_id(0)
        bb = tl.program_id(1)
        ob = bb * BN + tl.arange(0, BN)
        acc = tl.zeros((BN,), tl.float32)
        for k0 in range(0, K, BK):
            base = row * K + k0
            i = base + tl.arange(0, BK)
            w4 = tl.load(sm32_ptr + (base >> 2) + tl.arange(0, BK // 4))
            sm = tl.reshape(w4[:, None] >> ((tl.arange(0, 4) * 8)[None, :]), (BK,)) & 0xFF
            w8 = tl.load(nib32_ptr + (base >> 3) + tl.arange(0, BK // 8))
            nb = tl.reshape(w8[:, None] >> ((tl.arange(0, 8) * 4)[None, :]), (BK,)) & 0xF
            w32 = tl.load(hi32_ptr + (base >> 5) + tl.arange(0, BK // 32))
            hb = tl.reshape(w32[:, None] >> (tl.arange(0, 32)[None, :]), (BK,)) & 1
            idx = nb | (hb << 4)
            e = tl.load(lut_ptr + ((i >> CHUNK_SHIFT) << 5) + idx)
            u = ((sm >> 7) << 15) | (e << 7) | (sm & 0x7F)
            wv = u.to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)
            xt = tl.load(x_ptr + (k0 + tl.arange(0, BK))[:, None] * B + ob[None, :])
            acc += tl.sum(wv[:, None] * xt, 0)
        tl.store(y_ptr + row * B + ob, acc)

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - triton missing
    _HAVE_TRITON = False


def _pack_q13(w: torch.Tensor):
    """bf16 tensor -> (sm u8, nib u8, hi u8, luts i32 (n_chunks,32), n, shape).

    Bit-exact: unpack reproduces w's bf16 bit pattern exactly.
    """
    assert w.dtype == torch.bfloat16
    shape = w.shape
    u = w.reshape(-1).view(torch.int16).to(torch.int32) & 0xFFFF
    n = u.numel()
    pad = (-n) % 32
    if pad:
        u = torch.nn.functional.pad(u, (0, pad))
    exp = (u >> 7) & 0xFF
    sm = ((((u >> 15) & 1) << 7) | (u & 0x7F)).to(torch.uint8)
    idx = torch.empty_like(exp)
    luts = []
    for s in range(0, u.numel(), CHUNK):
        e = exp[s : s + CHUNK]
        uniq = torch.unique(e)
        if uniq.numel() > 32:
            raise ValueError("Q13: exponent band %d > 32 in chunk" % uniq.numel())
        lut = torch.zeros(32, dtype=torch.int32, device=u.device)
        lut[: uniq.numel()] = uniq
        inv = torch.zeros(256, dtype=torch.int32, device=u.device)
        inv[uniq.long()] = torch.arange(
            uniq.numel(), dtype=torch.int32, device=u.device
        )
        idx[s : s + CHUNK] = inv[e.long()]
        luts.append(lut)
    nib = ((idx[0::2] & 0xF) | ((idx[1::2] & 0xF) << 4)).to(torch.uint8)
    bits = (idx >> 4).reshape(-1, 8).to(torch.uint8)
    weights8 = (
        torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=u.device)
    )
    hi = (bits * weights8).sum(dim=1, dtype=torch.int32).to(torch.uint8)
    return sm, nib, hi, torch.stack(luts), n, shape


def _unpack_q13(sm, nib, hi, luts, n, shape):
    """Planes -> bf16 tensor, bit-exact."""
    total = sm.numel()
    idx = torch.empty(total, dtype=torch.int32, device=sm.device)
    nib32 = nib.to(torch.int32)
    idx[0::2] = nib32 & 0xF
    idx[1::2] = nib32 >> 4
    hb = hi.to(torch.int32)
    bit = (
        (hb.unsqueeze(1) >> torch.arange(8, dtype=torch.int32, device=sm.device)) & 1
    ).reshape(-1)
    idx = idx | (bit << 4)
    chunk_id = (
        torch.arange(total, dtype=torch.int64, device=sm.device) // CHUNK
    )
    exp = luts[chunk_id, idx.long()]
    sm32 = sm.to(torch.int32)
    u = ((sm32 >> 7) << 15) | (exp << 7) | (sm32 & 0x7F)
    w = u.to(torch.int16).view(torch.bfloat16)
    return w[:n].reshape(shape)


class Q13LinearMethod(UnquantizedLinearMethod):
    """Unquantized create_weights/loading; packs after load, fused apply."""

    packed_layers = 0
    packed_bytes_saved = 0

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w = layer.weight.data
        if w.dtype != torch.bfloat16:
            return  # leave non-bf16 layers untouched
        sm, nib, hi, luts, n, shape = _pack_q13(w)
        Q13LinearMethod.packed_layers += 1
        Q13LinearMethod.packed_bytes_saved += n * 2 - (
            sm.numel() + nib.numel() + hi.numel() + luts.numel() * 4
        )
        if Q13LinearMethod.packed_layers == 1:
            logger.info("Q13: packing engaged (first linear layer packed)")
        del layer._parameters["weight"]
        layer.register_buffer("q13_sm", sm, persistent=False)
        layer.register_buffer("q13_nib", nib, persistent=False)
        layer.register_buffer("q13_hi", hi, persistent=False)
        layer.register_buffer("q13_luts", luts, persistent=False)
        layer.q13_n = n
        layer.q13_shape = shape
        # int32 views for the fused kernels (padded to %32 so these are exact)
        layer.q13_sm32 = sm.view(torch.int32)
        layer.q13_nib32 = nib.view(torch.int32)
        layer.q13_hi32 = hi.view(torch.int32)
        layer.q13_lutflat = luts.reshape(-1).contiguous()
        M, Kd = shape
        layer.q13_fusable = (
            _HAVE_TRITON and _FUSED and Kd % 128 == 0 and (M * Kd) % 32 == 0
        )

    def _apply_fused(self, layer, x, bias):
        M, K = layer.q13_shape
        xs = x.reshape(-1, K)
        B = xs.shape[0]
        if B == 1:
            x32 = xs[0].to(torch.float32)
            y = torch.empty(M, dtype=torch.float32, device=x.device)
            _gemv_q13[(M,)](layer.q13_sm32, layer.q13_nib32, layer.q13_hi32,
                            layer.q13_lutflat, x32, y,
                            K=K, BK=128, CHUNK_SHIFT=CHUNK_SHIFT, num_warps=2)
            out = y.unsqueeze(0)
        else:
            BN = 32 if B >= 32 else 8
            Bp = ((B + BN - 1) // BN) * BN
            xT = torch.zeros(K, Bp, dtype=torch.float32, device=x.device)
            xT[:, :B] = xs.t().to(torch.float32)
            y = torch.empty(M * Bp, dtype=torch.float32, device=x.device)
            _gemm_q13[(M, Bp // BN)](layer.q13_sm32, layer.q13_nib32,
                                     layer.q13_hi32, layer.q13_lutflat, xT, y,
                                     K=K, B=Bp, BK=128, BN=BN,
                                     CHUNK_SHIFT=CHUNK_SHIFT, num_warps=2)
            out = y.reshape(M, Bp)[:, :B].t()
        out = out.to(x.dtype)
        if bias is not None:
            out = out + bias
        return out.reshape(*x.shape[:-1], M)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not hasattr(layer, "q13_sm"):  # non-bf16 layer: stock path
            return super().apply(layer, x, bias)
        if getattr(layer, "q13_fusable", False):
            return self._apply_fused(layer, x, bias)
        w = _unpack_q13(
            layer.q13_sm, layer.q13_nib, layer.q13_hi,
            layer.q13_luts, layer.q13_n, layer.q13_shape,
        )
        return self._gemm_impl(layer, x, w, bias)


@register_quantization_config("q13")
class Q13Config(QuantizationConfig):
    """Lossless 13-bit resident bf16 (CLI-only; no checkpoint config)."""

    @classmethod
    def get_name(cls) -> str:
        return "q13"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config) -> "Q13Config":
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            return Q13LinearMethod()
        return None
