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
round-tripped bit-exact). On Qwen2.5-Coder-7B (bf16, one B60), the fused
apply reads the planes directly inside the matmul (the expanded weight never
touches memory): batch-1 decode 29.5 vs 30.0 tok/s unquantized (0.98x), KV
cache 5.19 -> 7.26 GiB (+40% context), generated text identical. The fused
matmul accumulates in fp32 and rounds to bf16, so outputs sit within one bf16
ulp of the stock GEMM rather than bitwise equal; VLLM_Q13_FUSED=0 selects the
bit-exact unpack reference path (token-identical to unquantized, much
slower). Known gap: batched decode (2-31 concurrent sequences) runs ~0.4x
unquantized - the per-element plane unpack saturates the vector ALUs
alongside one batch column but not several, and the XMX units cannot be fed
from registers. Single-stream serving is the design point. Enable with
quantization="q13" after importing this module; registration is via the
decorator, no core files are touched.
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
# Fused apply is the default: validated in-engine per-call against the unpack
# reference on every layer shape (max one bf16 output ulp apart; see
# VLLM_Q13_CHECK) and greedy outputs matched bf16 on 0.5B and 7B smokes.
# VLLM_Q13_FUSED=0 falls back to unpack + the stock GEMM: bit-exact bf16
# matmul inputs (token-identical to unquantized), but it reconstructs every
# weight tensor each step, ~10-25x slower - a reference path, not a mode.
_FUSED = os.environ.get("VLLM_Q13_FUSED", "1") == "1"
# Forensics: VLLM_Q13_CHECK=1 runs the fused kernel AND the unpack reference
# on every fusable apply, logs each distinct (layer class, shape, batch)
# signature once with its fused-vs-reference error, and returns the reference
# so generation stays correct while the log maps where the fused glue breaks.
_CHECK = os.environ.get("VLLM_Q13_CHECK", "0") == "1"
# Forensics: VLLM_Q13_SYNC=1 synchronizes the device after every fused kernel
# launch. If fused output is garbage without it and clean with it, the bug is
# a stream/queue mismatch between the Triton launch and torch's ops, not math.
_SYNC = os.environ.get("VLLM_Q13_SYNC", "0") == "1"

try:
    import triton
    import triton.language as tl

    # v3 kernels: bf16 x in its native (B, K) row-major layout, bf16 y written
    # (B, M) row-major, B a RUNTIME arg with masked tail. The engine-side win
    # is host overhead: no transpose/pad/convert copies per apply (6 extra XPU
    # launches at v2) and no Triton recompile per distinct batch size.

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
            xv = tl.load(x_ptr + k0 + tl.arange(0, BK)).to(tl.float32)
            acc += wv * xv
        tl.store(y_ptr + row, tl.sum(acc, axis=0).to(tl.bfloat16))

    @triton.jit
    def _gemm_q13(sm32_ptr, nib32_ptr, hi32_ptr, lut_ptr, x_ptr, y_ptr,
                  B, M, K: tl.constexpr, BK: tl.constexpr,
                  BN: tl.constexpr, CHUNK_SHIFT: tl.constexpr):
        row = tl.program_id(0)
        bb = tl.program_id(1)
        ob = bb * BN + tl.arange(0, BN)
        bmask = ob < B
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
            xt = tl.load(x_ptr + ob[None, :] * K + (k0 + tl.arange(0, BK))[:, None],
                         mask=bmask[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(wv[:, None] * xt, 0)
        tl.store(y_ptr + ob * M + row, acc.to(tl.bfloat16), mask=bmask)

    @triton.jit
    def _gemm_q13_sb(sm32_ptr, nib32_ptr, hi32_ptr, lut_ptr, x_ptr, y_ptr,
                     B, M, K: tl.constexpr, BK: tl.constexpr,
                     BN: tl.constexpr, CHUNK_SHIFT: tl.constexpr):
        # small-batch GEMM: GEMV register shape (one (BK,) weight vector),
        # static loop over BN batch columns, scalar reduce per column folded
        # into a (BN,) acc. Best measured small-B form on 7B shapes - the
        # wide (BK, BN) tile variants go issue-bound and lose 3-5x.
        row = tl.program_id(0)
        bb = tl.program_id(1)
        ob = bb * BN + tl.arange(0, BN)
        acc = tl.zeros((BN,), tl.float32)
        ktrue = tl.arange(0, BK) < K  # all-true tensor, carries scalar masks
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
            for b in tl.static_range(BN):
                ob_b = bb * BN + b
                xb = tl.load(x_ptr + ob_b * K + k0 + tl.arange(0, BK),
                             mask=ktrue & (ob_b < B), other=0.0).to(tl.float32)
                s = tl.sum(wv * xb, 0)
                acc = tl.where(tl.arange(0, BN) == b, acc + s, acc)
        tl.store(y_ptr + ob * M + row, acc.to(tl.bfloat16), mask=ob < B)

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
            _HAVE_TRITON
            and (_FUSED or _CHECK)
            and Kd % 128 == 0
            and (M * Kd) % 32 == 0
        )

    def _apply_fused(self, layer, x, bias):
        M, K = layer.q13_shape
        xs = x.reshape(-1, K)
        if not xs.is_contiguous():
            xs = xs.contiguous()
        B = xs.shape[0]
        # y is written (B, M) row-major by the kernel itself: downstream
        # custom ops (silu_and_mul, attention) read raw pointers assuming
        # row-major, so the output must be truly contiguous, not a view.
        y = torch.empty(B, M, dtype=torch.bfloat16, device=x.device)
        if B == 1:
            # BK: largest tile dividing K; subgroup 16, one warp. Measured on
            # 7B tensors (200 reps): gate 1.21x / down 1.14x vs F.linear bf16
            # - the subgroup-32 default left ~10%, BK=128 another ~20%.
            BKv = 512 if K % 512 == 0 else (256 if K % 256 == 0 else 128)
            _gemv_q13[(M,)](layer.q13_sm32, layer.q13_nib32, layer.q13_hi32,
                            layer.q13_lutflat, xs, y,
                            K=K, BK=BKv, CHUNK_SHIFT=CHUNK_SHIFT,
                            num_warps=1, warp_size=16)
        elif B < 32:
            # small-batch decode: GEMV-shaped kernel, BK as in the GEMV
            BKv = 512 if K % 512 == 0 else (256 if K % 256 == 0 else 128)
            _gemm_q13_sb[(M, (B + 3) // 4)](
                layer.q13_sm32, layer.q13_nib32, layer.q13_hi32,
                layer.q13_lutflat, xs, y, B, M,
                K=K, BK=BKv, BN=4, CHUNK_SHIFT=CHUNK_SHIFT, num_warps=2)
        else:
            _gemm_q13[(M, (B + 31) // 32)](
                layer.q13_sm32, layer.q13_nib32, layer.q13_hi32,
                layer.q13_lutflat, xs, y, B, M,
                K=K, BK=128, BN=32, CHUNK_SHIFT=CHUNK_SHIFT, num_warps=2)
        if _SYNC:
            torch.xpu.synchronize()
        out = y if bias is None else y + bias
        return out.reshape(*x.shape[:-1], M)

    _check_seen: dict = {}

    def _apply_checked(self, layer, x, bias):
        fused = self._apply_fused(layer, x, bias)
        w = _unpack_q13(
            layer.q13_sm, layer.q13_nib, layer.q13_hi,
            layer.q13_luts, layer.q13_n, layer.q13_shape,
        )
        ref = self._gemm_impl(layer, x, w, bias)
        M, K = layer.q13_shape
        B = x.reshape(-1, K).shape[0]
        key = (type(layer).__name__, M, K, B)
        # compare EVERY call: a first-call-only compare hides launcher/JIT
        # bugs that appear once the compiled kernel is reused from cache
        d = (fused.float() - ref.float()).abs().max().item()
        s = ref.float().abs().max().clamp_min(1e-9).item()
        rel = d / s
        seen = Q13LinearMethod._check_seen
        prev, calls = seen.get(key, (-1.0, 0))
        seen[key] = (max(prev, rel), calls + 1)
        if prev < 0 or (rel > 1e-2 and rel > prev * 10):
            logger.info(
                "Q13 CHECK %s M=%d K=%d B=%d call#%d rel=%.2e%s | x dtype=%s "
                "contig=%s shape=%s stride=%s bias=%s fused_shape=%s",
                key[0], M, K, B, calls + 1, rel,
                " **DIVERGES**" if rel > 1e-2 else "",
                x.dtype, x.is_contiguous(), tuple(x.shape), tuple(x.stride()),
                "none" if bias is None else tuple(bias.shape),
                tuple(fused.shape),
            )
        return ref

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not hasattr(layer, "q13_sm"):  # non-bf16 layer: stock path
            return super().apply(layer, x, bias)
        if getattr(layer, "q13_fusable", False):
            if _CHECK:
                return self._apply_checked(layer, x, bias)
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
