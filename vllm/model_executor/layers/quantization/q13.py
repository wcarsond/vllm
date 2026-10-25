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
slower). Known gap: batched decode (2-31 concurrent sequences) runs below
unquantized - the per-element plane unpack saturates the vector ALUs
alongside one batch column but not several, and the XMX units cannot be fed
from registers. Single-stream serving is the design point. Enable with
quantization="q13" after importing this module; registration is via the
decorator, no core files are touched.

quantization="q12" is the occurrence-sorted variant and strictly dominates
when it applies: sort each chunk's exponent LUT by frequency and the top 16
cover 99.985% of weights (measured over all 7.62B weights of a 7B), so the
index needs only the 4-bit nibble - no fifth bit stored per weight. The
rare exponents file under a different pointer instead: a per-row exception
list carrying their exact values (~0.015% of weights, a few MB per model),
folded into the matmul as a CSR correction pass that costs nothing on the
99.9% of rows with no exceptions. Still bit-exact. 12.007 bits/weight,
r = 0.7504. Measured on the same 7B: weights 14.19 -> 11.22 GiB resident,
batch-1 decode 36.5 vs 30.0 tok/s unquantized (1.22x), KV cache +44%,
text identical.
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
# VLLM_Q13_HOSTLOAD=1: loader-pass pack - load the checkpoint into host
# memory, pack there, upload only the planes. See Q13LinearMethod docstring.
_HOSTLOAD = os.environ.get("VLLM_Q13_HOSTLOAD", "0") == "1"
# VLLM_Q13_PACKCACHE=1: packing is a pure function of the checkpoint, so do
# it once ever - planes are cached on disk keyed by a content fingerprint
# (a changed checkpoint misses and repacks; cache errors fall back to
# packing). Kills the pack compute on every start after the first.
_PACKCACHE = os.environ.get("VLLM_Q13_PACKCACHE", "0") == "1"
_PACKCACHE_DIR = os.environ.get(
    "VLLM_Q13_PACKCACHE_DIR", os.path.expanduser("~/.cache/vllm-q13")
)


def _fingerprint(w: torch.Tensor) -> str:
    import hashlib

    flat = w.reshape(-1)
    step = max(1, flat.numel() // 4096)
    sample = (flat[::step][:4096].contiguous().cpu()
              .view(torch.int16).numpy().tobytes())
    h = hashlib.blake2b(sample, digest_size=16)
    h.update(str((flat.numel(), tuple(w.shape))).encode())
    return h.hexdigest()

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

    # Q12 kernels: the occurrence-sorted variant. The per-chunk LUT holds the
    # 16 most frequent exponents (99.985% coverage measured model-wide on a
    # 7B), so the hi bitmap disappears from storage AND from the inner loop;
    # the rare exceptions ride a per-row CSR folded in after the K loop
    # (average well under one exception per row - most rows skip free).

    @triton.jit
    def _gemv_q12(sm32_ptr, nib32_ptr, lut_ptr, rowptr_ptr, exccol_ptr,
                  excdw_ptr, x_ptr, y_ptr, K: tl.constexpr, BK: tl.constexpr,
                  CHUNK_SHIFT: tl.constexpr, EXC_CAP: tl.constexpr):
        row = tl.program_id(0)
        acc = tl.zeros((BK,), tl.float32)
        for k0 in range(0, K, BK):
            base = row * K + k0
            i = base + tl.arange(0, BK)
            w4 = tl.load(sm32_ptr + (base >> 2) + tl.arange(0, BK // 4))
            sm = tl.reshape(w4[:, None] >> ((tl.arange(0, 4) * 8)[None, :]), (BK,)) & 0xFF
            w8 = tl.load(nib32_ptr + (base >> 3) + tl.arange(0, BK // 8))
            nb = tl.reshape(w8[:, None] >> ((tl.arange(0, 8) * 4)[None, :]), (BK,)) & 0xF
            e = tl.load(lut_ptr + ((i >> CHUNK_SHIFT) << 4) + nb)
            u = ((sm >> 7) << 15) | (e << 7) | (sm & 0x7F)
            wv = u.to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)
            xv = tl.load(x_ptr + k0 + tl.arange(0, BK)).to(tl.float32)
            acc += wv * xv
        s = tl.sum(acc, axis=0)
        r0 = tl.load(rowptr_ptr + row)
        r1 = tl.load(rowptr_ptr + row + 1)
        for jb in range(r0, r1, EXC_CAP):  # waves: any count, 0 = skip free
            jj = jb + tl.arange(0, EXC_CAP)
            m = jj < r1
            cols = tl.load(exccol_ptr + jj, mask=m, other=0)
            dws = tl.load(excdw_ptr + jj, mask=m, other=0.0)
            xe = tl.load(x_ptr + cols, mask=m, other=0.0).to(tl.float32)
            s += tl.sum(dws * xe, 0)
        tl.store(y_ptr + row, s.to(tl.bfloat16))

    @triton.jit
    def _gemm_q12_sb(sm32_ptr, nib32_ptr, lut_ptr, rowptr_ptr, exccol_ptr,
                     excdw_ptr, x_ptr, y_ptr, B, M,
                     K: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
                     CHUNK_SHIFT: tl.constexpr, EXC_CAP: tl.constexpr):
        row = tl.program_id(0)
        bb = tl.program_id(1)
        ob = bb * BN + tl.arange(0, BN)
        acc = tl.zeros((BN,), tl.float32)
        ktrue = tl.arange(0, BK) < K
        for k0 in range(0, K, BK):
            base = row * K + k0
            i = base + tl.arange(0, BK)
            w4 = tl.load(sm32_ptr + (base >> 2) + tl.arange(0, BK // 4))
            sm = tl.reshape(w4[:, None] >> ((tl.arange(0, 4) * 8)[None, :]), (BK,)) & 0xFF
            w8 = tl.load(nib32_ptr + (base >> 3) + tl.arange(0, BK // 8))
            nb = tl.reshape(w8[:, None] >> ((tl.arange(0, 8) * 4)[None, :]), (BK,)) & 0xF
            e = tl.load(lut_ptr + ((i >> CHUNK_SHIFT) << 4) + nb)
            u = ((sm >> 7) << 15) | (e << 7) | (sm & 0x7F)
            wv = u.to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)
            for b in tl.static_range(BN):
                ob_b = bb * BN + b
                xb = tl.load(x_ptr + ob_b * K + k0 + tl.arange(0, BK),
                             mask=ktrue & (ob_b < B), other=0.0).to(tl.float32)
                sb = tl.sum(wv * xb, 0)
                acc = tl.where(tl.arange(0, BN) == b, acc + sb, acc)
        r0 = tl.load(rowptr_ptr + row)
        r1 = tl.load(rowptr_ptr + row + 1)
        for jb in range(r0, r1, EXC_CAP):
            jj = jb + tl.arange(0, EXC_CAP)
            m = jj < r1
            cols = tl.load(exccol_ptr + jj, mask=m, other=0)
            dws = tl.load(excdw_ptr + jj, mask=m, other=0.0)
            for b in tl.static_range(BN):
                ob_b = bb * BN + b
                xe = tl.load(x_ptr + ob_b * K + cols, mask=m & (ob_b < B),
                             other=0.0).to(tl.float32)
                sb = tl.sum(dws * xe, 0)
                acc = tl.where(tl.arange(0, BN) == b, acc + sb, acc)
        tl.store(y_ptr + ob * M + row, acc.to(tl.bfloat16), mask=ob < B)

    @triton.jit
    def _gemm_q12(sm32_ptr, nib32_ptr, lut_ptr, rowptr_ptr, exccol_ptr,
                  excdw_ptr, x_ptr, y_ptr, B, M,
                  K: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
                  CHUNK_SHIFT: tl.constexpr, EXC_CAP: tl.constexpr):
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
            e = tl.load(lut_ptr + ((i >> CHUNK_SHIFT) << 4) + nb)
            u = ((sm >> 7) << 15) | (e << 7) | (sm & 0x7F)
            wv = u.to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)
            xt = tl.load(x_ptr + ob[None, :] * K + (k0 + tl.arange(0, BK))[:, None],
                         mask=bmask[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(wv[:, None] * xt, 0)
        r0 = tl.load(rowptr_ptr + row)
        r1 = tl.load(rowptr_ptr + row + 1)
        for jb in range(r0, r1, EXC_CAP):
            jj = jb + tl.arange(0, EXC_CAP)
            m = jj < r1
            cols = tl.load(exccol_ptr + jj, mask=m, other=0)
            dws = tl.load(excdw_ptr + jj, mask=m, other=0.0)
            xe = tl.load(x_ptr + ob[None, :] * K + cols[:, None],
                         mask=m[:, None] & bmask[None, :],
                         other=0.0).to(tl.float32)
            acc += tl.sum(dws[:, None] * xe, 0)
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


def _pack_q12(w: torch.Tensor):
    """bf16 tensor -> occurrence-sorted 12-bit planes + exact exceptions.

    Per 1M chunk the LUT holds the 16 MOST FREQUENT exponents; weights
    outside it (measured 0.015% model-wide on a 7B) are exceptions: their
    nibble says 0 and their exact bf16 bit pattern rides a side list.
    Returns (sm u8, nib u8, luts (n_chunks,16) i32, exc_gpos i64, exc_val
    i16, n, shape). Bit-exact: unpack scatters the exact values back.
    """
    assert w.dtype == torch.bfloat16
    shape = w.shape
    u = w.reshape(-1).view(torch.int16).to(torch.int32) & 0xFFFF
    n = u.numel()
    pad = (-n) % 32
    if pad:
        u = torch.nn.functional.pad(u, (0, pad))
    sm = ((((u >> 15) & 1) << 7) | (u & 0x7F)).to(torch.uint8)
    # fully batched per-chunk pack: chunks are independent, so histograms,
    # LUT ranking and index maps run as single batched ops instead of a
    # python loop (the loop packed a 7B in ~90 s on CPU; this is seconds).
    # Plane sizes keep the %32 padding; only the histogram view is
    # chunk-padded, transiently.
    dev = u.device
    L = u.numel()
    n_chunks = (L + CHUNK - 1) // CHUNK
    total = n_chunks * CHUNK
    ehist = u if total == L else torch.nn.functional.pad(u, (0, total - L))
    exp2d = ((ehist >> 7) & 0xFF).reshape(n_chunks, CHUNK).long()
    valid = (torch.arange(total, device=dev) < n).reshape(n_chunks, CHUNK)
    counts = torch.zeros(n_chunks, 256, dtype=torch.int32, device=dev)
    counts.scatter_add_(1, exp2d, valid.to(torch.int32))
    top = counts.argsort(dim=1, descending=True)[:, :16].contiguous()
    inv = torch.full((n_chunks, 256), -1, dtype=torch.int32, device=dev)
    inv.scatter_(1, top, torch.arange(16, dtype=torch.int32, device=dev)
                 .expand(n_chunks, 16).contiguous())
    idx = inv.gather(1, exp2d).reshape(-1)[:L]
    excmask = (idx < 0) & (torch.arange(L, device=dev) < n)
    gpos = torch.nonzero(excmask).reshape(-1).to(torch.int64)
    idx = idx.clamp_min(0)
    nib = ((idx[0::2] & 0xF) | ((idx[1::2] & 0xF) << 4)).to(torch.uint8)
    exc_val = u[gpos].to(torch.int16)
    return sm, nib, top.to(torch.int32), gpos, exc_val, n, shape


def _unpack_q12(sm, nib, luts, exc_gpos, exc_val, n, shape):
    """Planes + exception scatter -> bf16 tensor, bit-exact."""
    total = sm.numel()
    idx = torch.empty(total, dtype=torch.int32, device=sm.device)
    nib32 = nib.to(torch.int32)
    idx[0::2] = nib32 & 0xF
    idx[1::2] = nib32 >> 4
    chunk_id = torch.arange(total, dtype=torch.int64, device=sm.device) // CHUNK
    exp = luts[chunk_id, idx.long()]
    sm32 = sm.to(torch.int32)
    u = ((sm32 >> 7) << 15) | (exp << 7) | (sm32 & 0x7F)
    u[exc_gpos] = exc_val.to(torch.int32) & 0xFFFF
    w = u.to(torch.int16).view(torch.bfloat16)
    return w[:n].reshape(shape)


class Q13LinearMethod(UnquantizedLinearMethod):
    """Unquantized create_weights/loading; packs after load, fused apply.

    VLLM_Q13_HOSTLOAD=1 (loader-pass pack): linear weights are created on the
    CPU, the checkpoint loads into host memory, packing happens on the host,
    and only the 13-bit planes ever cross PCIe. Peak device memory during
    load drops from full bf16 to the packed size (the fit-cliff unlock: a
    model whose bf16 doesn't fit the card can still load packed), and the
    upload moves 0.8125x the bytes. requires_device_loading=False keeps the
    loader from moving the CPU weights to the device before we pack them.
    """

    packed_layers = 0
    packed_bytes_saved = 0

    def __init__(self, fmt: str = "q13"):
        super().__init__()
        self.fmt = fmt
        if _HOSTLOAD:
            self.requires_device_loading = False

    def create_weights(self, layer, *args, **kwargs):
        if not _HOSTLOAD:
            return super().create_weights(layer, *args, **kwargs)
        with torch.device("cpu"):
            return super().create_weights(layer, *args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w = layer.weight.data
        hostload = _HOSTLOAD and w.device.type == "cpu"
        if hostload:
            dev = torch.device("xpu", torch.xpu.current_device())
        if w.dtype != torch.bfloat16:
            if hostload:  # still bring the layer onto the card
                for pname, p in list(layer._parameters.items()):
                    if p is not None and p.device.type == "cpu":
                        p.data = p.data.to(dev)
            return  # leave non-bf16 layers unpacked
        M, Kd = w.shape
        fmt = self.fmt
        # pack cache: packing is a pure function of the checkpoint - a
        # content-fingerprint hit skips the pack compute entirely
        cpath = None
        cached = None
        if _PACKCACHE:
            try:
                cpath = os.path.join(
                    _PACKCACHE_DIR, f"{fmt}-{_fingerprint(w)}.pt"
                )
                if os.path.exists(cpath):
                    cached = torch.load(cpath, map_location="cpu")
            except Exception as e:  # cache can never break a load
                logger.warning("Q13 pack cache read failed (%s); packing", e)
                cached = None
        if cached is not None:
            fmt = cached["fmt"]
            n, shape = cached["n"], torch.Size(cached["shape"])
            tgt = w.device if not hostload else torch.device("cpu")
            sm, nib, luts = (cached[k].to(tgt) for k in ("sm", "nib", "luts"))
            if fmt == "q13":
                hi = cached["hi"].to(tgt)
            else:
                hi = None
                row_ptr, cols, dw, gpos_s, val_s = (
                    cached[k].to(tgt)
                    for k in ("row_ptr", "cols", "dw", "gpos", "val")
                )
        elif fmt == "q12":
            sm, nib, luts, gpos, exc_val, n, shape = _pack_q12(w)
            rows = (gpos // Kd).to(torch.int64)
            counts = torch.bincount(rows, minlength=M)
            order = torch.argsort(rows)
            gpos_s = gpos[order]
            val_s = exc_val[order]
            cols = (gpos_s % Kd).to(torch.int32)
            row_ptr = torch.zeros(M + 1, dtype=torch.int32, device=w.device)
            row_ptr[1:] = torch.cumsum(counts, 0).to(torch.int32)
            # dw = true value - what the main loop computes (lut[0] exp)
            smv = sm[gpos_s].to(torch.int32)
            lut0 = luts[(gpos_s // CHUNK).long(), 0]
            u_used = ((smv >> 7) << 15) | (lut0 << 7) | (smv & 0x7F)
            w_used = u_used.to(torch.int16).view(torch.bfloat16).to(torch.float32)
            w_true = val_s.view(torch.bfloat16).to(torch.float32)
            dw = (w_true - w_used).contiguous()
            if cols.numel() == 0:  # keep the kernels' pointers valid
                cols = torch.zeros(1, dtype=torch.int32, device=w.device)
                dw = torch.zeros(1, dtype=torch.float32, device=w.device)
            hi = None
        if cached is None and fmt == "q13":
            sm, nib, hi, luts, n, shape = _pack_q13(w)
        if cached is None and cpath is not None:
            try:
                os.makedirs(_PACKCACHE_DIR, exist_ok=True)
                d = {"fmt": fmt, "n": n, "shape": tuple(shape),
                     "sm": sm.cpu(), "nib": nib.cpu(), "luts": luts.cpu()}
                if fmt == "q13":
                    d["hi"] = hi.cpu()
                else:
                    d.update(row_ptr=row_ptr.cpu(), cols=cols.cpu(),
                             dw=dw.cpu(), gpos=gpos_s.cpu(), val=val_s.cpu())
                torch.save(d, cpath)
            except Exception as e:
                logger.warning("Q13 pack cache write failed (%s)", e)
        if hostload:
            sm, nib, luts = sm.to(dev), nib.to(dev), luts.to(dev)
            if fmt == "q13":
                hi = hi.to(dev)
            else:
                row_ptr, cols, dw = row_ptr.to(dev), cols.to(dev), dw.to(dev)
                gpos_s, val_s = gpos_s.to(dev), val_s.to(dev)
        Q13LinearMethod.packed_layers += 1
        side = (hi.numel() if fmt == "q13" else
                row_ptr.numel() * 4 + cols.numel() * 8 + gpos_s.numel() * 10)
        Q13LinearMethod.packed_bytes_saved += n * 2 - (
            sm.numel() + nib.numel() + luts.numel() * 4 + side
        )
        if Q13LinearMethod.packed_layers == 1:
            logger.info("Q13: packing engaged (first linear layer packed, fmt=%s)", fmt)
        del layer._parameters["weight"]
        if hostload:  # bias (and any other host params) follow the planes
            for pname, p in list(layer._parameters.items()):
                if p is not None and p.device.type == "cpu":
                    p.data = p.data.to(dev)
        layer.q13_fmt = fmt
        layer.register_buffer("q13_sm", sm, persistent=False)
        layer.register_buffer("q13_nib", nib, persistent=False)
        layer.register_buffer("q13_luts", luts, persistent=False)
        if fmt == "q13":
            layer.register_buffer("q13_hi", hi, persistent=False)
            layer.q13_hi32 = hi.view(torch.int32)
        else:
            layer.register_buffer("q12_rowptr", row_ptr, persistent=False)
            layer.register_buffer("q12_exccol", cols, persistent=False)
            layer.register_buffer("q12_excdw", dw, persistent=False)
            layer.register_buffer("q12_gpos", gpos_s, persistent=False)
            layer.register_buffer("q12_excval", val_s, persistent=False)
        layer.q13_n = n
        layer.q13_shape = shape
        # int32 views for the fused kernels (padded to %32 so these are exact)
        layer.q13_sm32 = sm.view(torch.int32)
        layer.q13_nib32 = nib.view(torch.int32)
        layer.q13_lutflat = luts.reshape(-1).contiguous()
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
        # BK: largest tile dividing K; subgroup 16, one warp on the GEMV.
        # Measured on 7B tensors (200 reps): gate 1.21x / down 1.14x vs
        # F.linear bf16 - subgroup-32 left ~10%, BK=128 another ~20%.
        BKv = 512 if K % 512 == 0 else (256 if K % 256 == 0 else 128)
        if getattr(layer, "q13_fmt", "q13") == "q12":
            args = (layer.q13_sm32, layer.q13_nib32, layer.q13_lutflat,
                    layer.q12_rowptr, layer.q12_exccol, layer.q12_excdw)
            if B == 1:
                _gemv_q12[(M,)](*args, xs, y, K=K, BK=BKv,
                                CHUNK_SHIFT=CHUNK_SHIFT, EXC_CAP=64,
                                num_warps=1, warp_size=16)
            elif B < 32:
                _gemm_q12_sb[(M, (B + 3) // 4)](*args, xs, y, B, M,
                                                K=K, BK=BKv, BN=4,
                                                CHUNK_SHIFT=CHUNK_SHIFT,
                                                EXC_CAP=64, num_warps=2)
            else:
                _gemm_q12[(M, (B + 31) // 32)](*args, xs, y, B, M,
                                               K=K, BK=128, BN=32,
                                               CHUNK_SHIFT=CHUNK_SHIFT,
                                               EXC_CAP=64, num_warps=2)
        elif B == 1:
            _gemv_q13[(M,)](layer.q13_sm32, layer.q13_nib32, layer.q13_hi32,
                            layer.q13_lutflat, xs, y,
                            K=K, BK=BKv, CHUNK_SHIFT=CHUNK_SHIFT,
                            num_warps=1, warp_size=16)
        elif B < 32:
            # small-batch decode: GEMV-shaped kernel, BK as in the GEMV
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

    def _unpack_layer(self, layer):
        if getattr(layer, "q13_fmt", "q13") == "q12":
            return _unpack_q12(
                layer.q13_sm, layer.q13_nib, layer.q13_luts,
                layer.q12_gpos, layer.q12_excval,
                layer.q13_n, layer.q13_shape,
            )
        return _unpack_q13(
            layer.q13_sm, layer.q13_nib, layer.q13_hi,
            layer.q13_luts, layer.q13_n, layer.q13_shape,
        )

    def _apply_checked(self, layer, x, bias):
        fused = self._apply_fused(layer, x, bias)
        w = self._unpack_layer(layer)
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
        w = self._unpack_layer(layer)
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


@register_quantization_config("q12")
class Q12Config(Q13Config):
    """Occurrence-sorted 12-bit resident bf16: per-chunk LUT of the 16 most
    frequent exponents (99.985% coverage measured on a 7B), rare exponents as
    exact per-row exceptions. r = 0.7504 vs Q13's 0.8125; still lossless."""

    @classmethod
    def get_name(cls) -> str:
        return "q12"

    @classmethod
    def from_config(cls, config) -> "Q12Config":
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            return Q13LinearMethod(fmt="q12")
        return None
