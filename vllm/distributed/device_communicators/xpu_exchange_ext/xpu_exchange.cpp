// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// xpu_exchange.cpp - two-card all_reduce for vLLM on Windows XPU that never blocks the Python thread.
// Built by kernels\build-exchange.input.cmd into _xpu_exchange.pyd beside xpu_gpu_exchange.py; see that file for the
// Python side and PORT-NOTES.md ("GPU-TIMELINE EXCHANGE", 2026-10-11) for the measurements behind every choice here.
//
// The two B60s have no peer access and this driver lets no kernel touch another process's memory, so host RAM is the
// channel and each GPU only ever touches memory of its own process. Per all_reduce (sequence s, buffer b = s & 1):
//   main thread (torch's current XPU stream, in order, nothing waits):
//     K1  the card writes its half into mine[b] (pinned host USM) as tagged words: 4 bytes of data + the 4-byte tag s
//     post {s, bytes} to the helper thread; wait (host side) until the peer has published s-1
//     K2  every lane re-reads its tagged words of theirs[b] (host USM) until the tag is s, then adds the data in place
//   helper thread, per job: read mine[b] until every tag is s (invalidating its cache lines first); memcpy the data into
//     my shared slot; publish s in my shared flag; spin until the peer's shared flag >= s; write the peer's slot into
//     theirs[b] as tagged words.
// Tensors over 64 KB (prefill chunks) take the big path instead: the copy engine moves the plain bytes into pinned host
// memory, a one-lane kernel (in order, so after the copy) stores s into a marker line, the helper waits for the marker
// (invalidating its line and the data's lines before reading), hops the bytes, writes the peer's half into pinned memory
// and stores a second marker; on the card a one-lane kernel polls that marker, then the copy engine brings the bytes
// into card memory and a plain kernel adds them. Tagging 3.7M words with 8-byte stores took ~8 ms; this takes ~1.5.
// Why tags on every word (the NCCL "LL" protocol) instead of a flag: on this driver a flag can be seen before the data it
// guards, in both directions. The card's stores into host memory do not invalidate the CPU's cached lines, and the card
// kept serving a line of host memory from its own cache after the CPU had rewritten it (probe/cache-coherence.cpp,
// 2026-10-11). With a tag in the same 8 bytes as the data, a stale line fails its tag and is simply read again; the
// re-read is a cache-bypassing load followed by an acquire fence, the one pattern that reliably sees a CPU write from a
// running kernel (test 9: 20 of 20 within ~12 us). Bytes on the wire double (7 KB -> 14 KB per decode all_reduce), which
// is nothing against the 258 us the host-synchronous Python exchange cost.
// Double buffering by parity is safe because the compute queue is in order: job s+2 cannot start before K2(s) has run.
// Failure policy: a lane that polls past SPIN_LIMIT, or a helper wait over 60 s, records the sequence number in
// host-visible status words and every later all_reduce_ raises, so a lost peer ends the run loudly rather than letting
// a stale half into the sum.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <torch/extension.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace {

constexpr size_t HEADER = 4096;
constexpr size_t FLAG_OFF[2] = {0, 64};        // shared: per-rank published sequence, one cache line each
constexpr int WG = 512;
constexpr size_t SMALL = 64u << 10;            // one work-group per kernel up to here (data bytes)
constexpr uint32_t SPIN_LIMIT = 6000000;       // polls per lane before it gives up: ~1 s at 150-180 ns a poll
                                               // (probe/poll-rate.cpp), under the 2 s Windows TDR default (TdrDelay unset)
// How long the host side waits for the peer before calling it gone. 600 s by default (VLLM_XPU_EXCHANGE_TIMEOUT_S):
// the two workers compile their graphs independently when torch.compile is on, and one can be minutes ahead of the
// other (11:30: rank 0 waited at all_reduce #64 while rank 1 was still in Inductor; 60 s was not enough).
inline double timeout_s() {
  const char* e = std::getenv("VLLM_XPU_EXCHANGE_TIMEOUT_S");
  if (e && *e) { double v = std::atof(e); if (v > 0) return v; }
  return 600.0;
}
constexpr int RING = 1024;

namespace iex = sycl::ext::intel::experimental;
namespace oex = sycl::ext::oneapi::experimental;
using uncached_wprops = decltype(oex::properties(
    iex::write_hint<iex::cache_control<iex::cache_mode::uncached, oex::cache_level::L1, oex::cache_level::L3>>));
template <class T> using uncached_wptr = oex::annotated_ptr<T, uncached_wprops>;
using uncached_rprops = decltype(oex::properties(
    iex::read_hint<iex::cache_control<iex::cache_mode::uncached, oex::cache_level::L1, oex::cache_level::L2, oex::cache_level::L3>>));
template <class T> using uncached_rptr = oex::annotated_ptr<T, uncached_rprops>;

enum DType : int { F32 = 0, BF16 = 1, F16 = 2 };

inline uint64_t tagged(uint32_t data, uint32_t tag) { return (uint64_t)data | ((uint64_t)tag << 32); }
inline uint32_t data_of(uint64_t w) { return (uint32_t)w; }
inline uint32_t tag_of(uint64_t w) { return (uint32_t)(w >> 32); }

// Add two 32-bit words holding one float or two 16-bit values, rounding exactly as torch does on the host (float math,
// round to nearest even), so both ranks hold identical bytes.
template <int DT> inline uint32_t add_words(uint32_t a, uint32_t b) {
  if constexpr (DT == F32) {
    return sycl::bit_cast<uint32_t>(sycl::bit_cast<float>(a) + sycl::bit_cast<float>(b));
  } else if constexpr (DT == BF16) {
    using bf16 = sycl::ext::oneapi::bfloat16;
    float lo = float(sycl::bit_cast<bf16>((uint16_t)(a & 0xffffu))) + float(sycl::bit_cast<bf16>((uint16_t)(b & 0xffffu)));
    float hi = float(sycl::bit_cast<bf16>((uint16_t)(a >> 16))) + float(sycl::bit_cast<bf16>((uint16_t)(b >> 16)));
    return (uint32_t)sycl::bit_cast<uint16_t>(bf16(lo)) | ((uint32_t)sycl::bit_cast<uint16_t>(bf16(hi)) << 16);
  } else {
    float lo = float(sycl::bit_cast<sycl::half>((uint16_t)(a & 0xffffu))) + float(sycl::bit_cast<sycl::half>((uint16_t)(b & 0xffffu)));
    float hi = float(sycl::bit_cast<sycl::half>((uint16_t)(a >> 16))) + float(sycl::bit_cast<sycl::half>((uint16_t)(b >> 16)));
    return (uint32_t)sycl::bit_cast<uint16_t>(sycl::half(lo)) | ((uint32_t)sycl::bit_cast<uint16_t>(sycl::half(hi)) << 16);
  }
}

// one lane: re-read a tagged word (cache-bypassing load + acquire fence) until its tag is `tag`; returns the word, or
// records the failure and returns the last value read
inline uint64_t wait_tagged(const uint64_t* p, uint32_t tag, int64_t seq, int64_t* status) {
  uncached_rptr<const uint64_t> r{p};
  uint32_t polls = 0;
  for (;;) {
    uint64_t v = r[0];
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    if (tag_of(v) == tag) return v;
    if (++polls > SPIN_LIMIT) { uncached_wptr<int64_t> s{status}; if (s[0] == 0) s[0] = seq; return v; }  // first failure wins
  }
}
// the same for four consecutive tagged words in one 32-byte load (a quarter of the PCIe transactions: each 8-byte
// cache-bypassing load costs ~1 us of latency and the card keeps few of them in flight)
using u64x4 = sycl::vec<uint64_t, 4>;
inline u64x4 wait_tagged4(const uint64_t* p, uint32_t tag, int64_t seq, int64_t* status) {
  uncached_rptr<const u64x4> r{reinterpret_cast<const u64x4*>(p)};
  uint32_t polls = 0;
  for (;;) {
    u64x4 v = r[0];
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    if (tag_of(v[0]) == tag && tag_of(v[1]) == tag && tag_of(v[2]) == tag && tag_of(v[3]) == tag) return v;
    if (++polls > SPIN_LIMIT) { uncached_wptr<int64_t> s{status}; if (s[0] == 0) s[0] = seq; return v; }
  }
}

struct Job { int64_t seq; size_t bytes; };
// per-exchange timings taken by the helper (microseconds), kept for the last STATS jobs
struct Stat { int64_t seq; size_t bytes; double wait_done, memcpy_in, wait_peer, memcpy_out, enqueue; };
constexpr int STATS = 64;

// Drop this core's cached copies of [p, p + bytes) (clflushopt: pipelined, ~1-2 ns a line; Arrow Lake has it). The
// card's stores into host memory do not invalidate CPU cache lines (probe test 8), so the helper does it before reading.
__attribute__((target("clflushopt"))) static void invalidate_lines(const void* p, size_t bytes) {
  const char* c = static_cast<const char*>(p);
  for (size_t off = 0; off < bytes; off += 64) _mm_clflushopt(const_cast<char*>(c) + off);
  _mm_sfence();
}

// Direct big path (2026-10-13): import the shared section into this process's Level Zero driver(s) with
// zexDriverImportExternalPointer, so the card's COPY ENGINE can read and write the section itself. Kernels still fault on
// imported memory (probe/import-test, 2026-10-11), but the big path only ever touches it with copies. probe/section-copy
// (two processes, one section, both cards): 14.7 MB all_reduce round 1,403 us, every sum bit-exact, against 4,204 us for
// the helper-memcpy path and ~2,720 us for oneCCL on Linux; without the import the same copies run at half speed.
// Returns the driver and release function on success; VLLM_XPU_EXCHANGE_DIRECT=0 turns it off.
typedef struct _ze_driver_handle_t* ze_driver_handle_t;
typedef uint32_t ze_result_t;
typedef ze_result_t (*pfn_zeInit)(uint32_t);
typedef ze_result_t (*pfn_zeDriverGet)(uint32_t*, ze_driver_handle_t*);
typedef ze_result_t (*pfn_zeDriverGetExtensionFunctionAddress)(ze_driver_handle_t, const char*, void**);
typedef ze_result_t (*pfn_zexDriverImportExternalPointer)(ze_driver_handle_t, void*, size_t);
typedef ze_result_t (*pfn_zexDriverReleaseImportedPointer)(ze_driver_handle_t, void*);
struct Imported { std::vector<std::pair<ze_driver_handle_t, pfn_zexDriverReleaseImportedPointer>> drivers; };
static bool import_for_copy_engine(void* p, size_t bytes, Imported& out) {
  const char* e = std::getenv("VLLM_XPU_EXCHANGE_DIRECT");
  if (e && std::string(e) == "0") return false;
  HMODULE lib = LoadLibraryA("ze_loader.dll");
  if (!lib) return false;
  auto zeInit = (pfn_zeInit)GetProcAddress(lib, "zeInit");
  auto zeDriverGet = (pfn_zeDriverGet)GetProcAddress(lib, "zeDriverGet");
  auto getExt = (pfn_zeDriverGetExtensionFunctionAddress)GetProcAddress(lib, "zeDriverGetExtensionFunctionAddress");
  if (!zeInit || !zeDriverGet || !getExt) return false;
  zeInit(0);
  uint32_t n = 0;
  if (zeDriverGet(&n, nullptr) != 0 || n == 0) return false;
  std::vector<ze_driver_handle_t> drivers(n);
  zeDriverGet(&n, drivers.data());
  // the loader lists every Level Zero driver (the NPU has one too): import into each that offers the extension, as the
  // probe did, so the GPU's driver is certainly among them
  for (uint32_t d = 0; d < n; ++d) {
    void* fi = nullptr; void* fr = nullptr;
    getExt(drivers[d], "zexDriverImportExternalPointer", &fi);
    getExt(drivers[d], "zexDriverReleaseImportedPointer", &fr);
    if (!fi) continue;
    if (((pfn_zexDriverImportExternalPointer)fi)(drivers[d], p, bytes) == 0)
      out.drivers.emplace_back(drivers[d], (pfn_zexDriverReleaseImportedPointer)fr);
  }
  return !out.drivers.empty();
}

class Exchange {
 public:
  Exchange(int rank, int world_size, const std::string& name, size_t slot_bytes, int device_index)
      : rank_(rank), peer_(1 - rank), slot_bytes_(slot_bytes), device_index_(device_index) {
    if (world_size != 2) throw std::invalid_argument("xpu exchange is written for two cards");
    const size_t total = HEADER + 4 * slot_bytes;
    if (rank == 0) {
      hmap_ = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, (DWORD)(total >> 32), (DWORD)(total & 0xffffffffu), name.c_str());
    } else {
      for (int i = 0; i < 600 && !hmap_; ++i) { hmap_ = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, name.c_str()); if (!hmap_) Sleep(50); }
    }
    if (!hmap_) throw std::runtime_error("xpu exchange: shared mapping '" + name + "' failed, error " + std::to_string(GetLastError()));
    base_ = reinterpret_cast<char*>(MapViewOfFile(hmap_, FILE_MAP_ALL_ACCESS, 0, 0, total));
    if (!base_) throw std::runtime_error("xpu exchange: MapViewOfFile failed, error " + std::to_string(GetLastError()));
    if (rank == 0) std::memset(base_, 0, HEADER);   // flags; the slots need no zeroing
    total_ = total;
    direct_ = import_for_copy_engine(base_, total, imported_);
    { const char* f = std::getenv("VLLM_XPU_EXCHANGE_FUSED"); fused_ = f && std::string(f) == "1"; }
    { const char* a = std::getenv("VLLM_XPU_EXCHANGE_AHEAD"); int k = a ? std::atoi(a) : 1; ahead_ = k < 1 ? 1 : (k > 64 ? 64 : k); }
    sycl::context& ctx = c10::xpu::get_device_context();
    sycl::device& dev = c10::xpu::get_raw_device(device_index);
    for (int b = 0; b < 2; ++b) {
      // small path: tagged words, 8 bytes per data word, up to SMALL data bytes
      mine_[b] = static_cast<uint64_t*>(sycl::malloc_host(2 * SMALL, ctx));
      theirs_[b] = static_cast<uint64_t*>(sycl::malloc_host(2 * SMALL, ctx));
      // big path: plain bytes moved by the copy engine, a tagged marker word for the handover; with the section
      // imported (direct_) the copy engine uses the shared slots themselves and no private staging is needed
      if (!direct_) {
        mine_plain_[b] = static_cast<char*>(sycl::malloc_host(slot_bytes, ctx));
        theirs_plain_[b] = static_cast<char*>(sycl::malloc_host(slot_bytes, ctx));
      }
      theirs_dev_[b] = static_cast<char*>(sycl::malloc_device(slot_bytes, dev, ctx));
    }
    status_ = static_cast<int64_t*>(sycl::malloc_host(2 * 64, ctx));
    marker_mine_ = static_cast<int64_t*>(sycl::malloc_host(2 * 64, ctx));
    marker_theirs_ = static_cast<int64_t*>(sycl::malloc_host(2 * 64, ctx));
    for (int b = 0; b < 2; ++b)
      if (!mine_[b] || !theirs_[b] || (!direct_ && (!mine_plain_[b] || !theirs_plain_[b])) || !theirs_dev_[b])
        throw std::runtime_error("xpu exchange: USM allocation failed");
    if (!status_ || !marker_mine_ || !marker_theirs_) throw std::runtime_error("xpu exchange: USM allocation failed");
    for (int b = 0; b < 2; ++b) { std::memset(mine_[b], 0, 2 * SMALL); std::memset(theirs_[b], 0, 2 * SMALL); }
    std::memset(status_, 0, 2 * 64); std::memset(marker_mine_, 0, 2 * 64); std::memset(marker_theirs_, 0, 2 * 64);
    helper_ = std::thread(&Exchange::helper_main, this);
  }

  ~Exchange() { close(); }

  // The hot path: no tensor when this one is not for the exchange (dtype, size, layout) so the caller can fall back,
  // without a second round of Python checks per layer; raises if an earlier exchange failed.
  // Out of place on purpose: vLLM registers all_reduce as a functional custom op (the fake returns empty_like), so under
  // torch.compile Inductor reuses the input's buffer right after the call; an in-place result was overwritten by the next
  // kernel and the compiled two-card run produced garbage (2026-10-11 12:00). The add kernel writes a fresh tensor, no
  // extra launch.
  std::optional<at::Tensor> try_all_reduce_(const at::Tensor& t) {
    if (!t.is_xpu() || !t.is_contiguous()) return std::nullopt;
    const size_t bytes = (size_t)t.numel() * t.element_size();
    if (bytes > slot_bytes_ || bytes % 4 != 0) return std::nullopt;
    int dt;
    switch (t.scalar_type()) {
      case at::kFloat: dt = F32; break;
      case at::kBFloat16: dt = BF16; break;
      case at::kHalf: dt = F16; break;
      default: return std::nullopt;
    }
    return run(t, bytes, dt);
  }

  at::Tensor all_reduce_(const at::Tensor& t) {
    TORCH_CHECK(t.is_xpu(), "xpu exchange: tensor must live on an XPU");
    TORCH_CHECK(t.is_contiguous(), "xpu exchange: tensor must be contiguous");
    const size_t bytes = (size_t)t.numel() * t.element_size();
    TORCH_CHECK(bytes <= slot_bytes_, "xpu exchange: tensor larger than a slot");
    TORCH_CHECK(bytes % 4 == 0, "xpu exchange: byte count must be a multiple of 4");
    int dt;
    switch (t.scalar_type()) {
      case at::kFloat: dt = F32; break;
      case at::kBFloat16: dt = BF16; break;
      case at::kHalf: dt = F16; break;
      default: TORCH_CHECK(false, "xpu exchange: unsupported dtype ", t.scalar_type());
    }
    return run(t, bytes, dt);
  }

 private:
  at::Tensor run(const at::Tensor& t, size_t bytes, int dt) {
    check_status();
    at::Tensor out = at::empty_like(t);
    if (bytes == 0) return out;
    sycl::queue& q = c10::xpu::getCurrentXPUStream(t.device().index()).queue();
    const int64_t seq = ++seq_;
    const uint32_t tag = (uint32_t)seq;
    const int b = (int)(seq & 1);
    uint32_t* x = static_cast<uint32_t*>(t.data_ptr());
    uint32_t* y = static_cast<uint32_t*>(out.data_ptr());
    const size_t nwords = bytes / 4;
    if (bytes <= SMALL) ++n_small_; else ++n_big_;
    if (bytes <= SMALL) {
      // Small path, two launches: the export goes out *before* the guard so the helper can start the hop while the
      // guard runs (one merged kernel launched after the guard measured 46.0 against 47.1 tok/s, 11:40). The export
      // has no barrier: the helper validates every word's tag. The 8-byte stores stay scalar: a 32-byte vector store
      // with the uncached hint ran ~80x slower (14.7 MB in 400 ms instead of 5 ms; 10:15).
      uint64_t* mine = mine_[b];
      const uint64_t* theirs = theirs_[b];
      int64_t* status = status_;
      if (fused_) {
        // One kernel per all_reduce (VLLM_XPU_EXCHANGE_FUSED=1, 2026-10-13), launched BEFORE the guard as the export
        // was: every lane writes its tagged words, then waits for the peer's and adds. Saves one launch and one
        // dispatch gap per exchange (Linux's oneCCL LL256 does it in one kernel: 13 us against our 28). All 512 lanes are
        // one work-group, resident together, so every export is written before any lane can block on its poll. The
        // earlier fused try (11:40, 2026-10-11) launched AFTER the guard and lost: the export waited on the host.
        // MEASURED 2026-10-13: no gain (7 KB back to back 28.3 us fused vs 27.8 two kernels, both bit-exact), so the
        // launch count is not the cost; the round trip through the two CPU helpers is. Kept off by default, for the
        // record. probe/one-process-pingpong: two cards in ONE process polling one host buffer do 7 KB in 14.3 us one
        // way, i.e. a helper-free exchange would match oneCCL's 13 us, but only with a single-process two-card executor.
        switch (dt) {
          case F32: fused_kernel<F32>(q, x, y, mine, theirs, status, seq, tag, nwords); break;
          case BF16: fused_kernel<BF16>(q, x, y, mine, theirs, status, seq, tag, nwords); break;
          default: fused_kernel<F16>(q, x, y, mine, theirs, status, seq, tag, nwords); break;
        }
        post(seq, bytes);
        guard(seq);
        return out;
      }
      q.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
        uncached_wptr<uint64_t> m{mine};
        for (size_t i = it.get_local_id(0); i < nwords; i += WG) m[i] = tagged(x[i], tag);
      });
      post(seq, bytes);
      guard(seq);
      switch (dt) {
        case F32: add_kernel<F32>(q, x, y, theirs, status, seq, tag, nwords); break;
        case BF16: add_kernel<BF16>(q, x, y, theirs, status, seq, tag, nwords); break;
        default: add_kernel<F16>(q, x, y, theirs, status, seq, tag, nwords); break;
      }
      return out;
    } else {
      // Big path (prefill chunks, MBs): the copy engine moves the bytes into pinned host memory at PCIe speed, then a
      // one-lane kernel, which the in-order queue starts only after the copy has completed, stores the sequence into
      // the marker line. Tagging every word of a 14.7 MB tensor with 8-byte stores took ~8 ms; this takes ~1.5.
      // direct_: straight into my slot of the shared section (imported for the copy engine), no helper memcpy later
      q.memcpy(direct_ ? slot(rank_, b) : mine_plain_[b], x, bytes);
      int64_t* marker = marker_mine_ + b * 8;
      q.single_task([=]() { uncached_wptr<int64_t> m{marker}; m[0] = seq; });
    }
    post(seq, bytes);
    guard(seq);
    int64_t* status = status_;
    {
      // K2a: one lane waits for the helper's marker (the peer's half is complete in theirs_plain[b] and flushed to
      // memory); then the copy engine brings it into card memory; then K2b adds from card memory (a fresh kernel,
      // plain loads: the coherence probe found copies into card memory visible to the next kernel in every case).
      const int64_t* marker = marker_theirs_ + b * 8;
      q.single_task([=]() {
        uncached_rptr<const int64_t> r{marker};
        uint32_t polls = 0;
        for (;;) {
          int64_t v = r[0];
          sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
          if (v >= seq) break;
          if (++polls > SPIN_LIMIT) { uncached_wptr<int64_t> s{status}; if (s[0] == 0) s[0] = seq; break; }
        }
      });
      // direct_: straight out of the peer's slot. Reuse is safe: my copy into slot b at s+2 runs after my poll at s+1,
      // which needs the peer's flag s+1, which the peer publishes only after its card has read my slot b at s (its queue
      // is in order: copy-out s, add s, copy-in s+1, marker s+1).
      q.memcpy(theirs_dev_[b], direct_ ? slot(peer_, b) : theirs_plain_[b], bytes);
      const uint32_t* theirs = reinterpret_cast<const uint32_t*>(theirs_dev_[b]);
      switch (dt) {
        case F32: q.parallel_for(sycl::range<1>(nwords), [=](sycl::id<1> i) { y[i] = add_words<F32>(x[i], theirs[i]); }); break;
        case BF16: q.parallel_for(sycl::range<1>(nwords), [=](sycl::id<1> i) { y[i] = add_words<BF16>(x[i], theirs[i]); }); break;
        default: q.parallel_for(sycl::range<1>(nwords), [=](sycl::id<1> i) { y[i] = add_words<F16>(x[i], theirs[i]); }); break;
      }
    }
    return out;
  }

 public:
  bool direct() const { return direct_; }
  // (gpu_timeout_seq, helper_timeout_seq, sequence): zero means no failure so far
  std::tuple<int64_t, int64_t, int64_t> status() const {
    return {reinterpret_cast<volatile int64_t*>(status_)[0], reinterpret_cast<volatile int64_t*>(status_)[8], seq_};
  }

  // Everything the two sides have said to each other, for the log when an all_reduce fails.
  std::string debug() {
    volatile int64_t* st = reinterpret_cast<volatile int64_t*>(status_);
    volatile int64_t* my_flag = reinterpret_cast<volatile int64_t*>(base_ + FLAG_OFF[rank_]);
    volatile int64_t* peer_flag = reinterpret_cast<volatile int64_t*>(base_ + FLAG_OFF[peer_]);
    std::string s = "rank " + std::to_string(rank_) + ": seq " + std::to_string(seq_) +
        ", gpu_timeout_at " + std::to_string(st[0]) + ", helper_timeout_at " + std::to_string(st[8]) +
        ", my_flag " + std::to_string(*my_flag) + ", peer_flag " + std::to_string(*peer_flag) +
        ", helper head " + std::to_string(head_.load()) + " tail " + std::to_string(tail_.load()) +
        ", small/big calls " + std::to_string(n_small_) + "/" + std::to_string(n_big_) + ", last jobs:";
    int64_t h = head_.load();
    for (int64_t i = std::max<int64_t>(0, h - 6); i < h; ++i)
      s += " (" + std::to_string(ring_[i % RING].seq) + ", " + std::to_string(ring_[i % RING].bytes) + " B)";
    auto words = [](const uint64_t* p) {
      char buf[80];
      std::snprintf(buf, sizeof buf, "%08x@%u %08x@%u", data_of(p[0]), tag_of(p[0]), data_of(p[1]), tag_of(p[1]));
      return std::string(buf);
    };
    auto plain = [](const char* p) {
      char buf[40];
      const uint32_t* w = reinterpret_cast<const uint32_t*>(p);
      std::snprintf(buf, sizeof buf, "%08x %08x", w[0], w[1]);
      return std::string(buf);
    };
    for (int b = 0; b < 2; ++b) {
      s += "; buf" + std::to_string(b) + " mine=" + words(mine_[b]) + " myslot=" + plain(slot(rank_, b)) +
           " peerslot=" + plain(slot(peer_, b)) + " theirs=" + words(theirs_[b]);
    }
    s += "; helper stage times in us for the last jobs (wait_for_card, memcpy_in, wait_for_peer, write_out, -):";
    int64_t tl = tail_.load();
    for (int64_t i = std::max<int64_t>(1, tl - 5); i <= tl && i >= 1; ++i) {
      const Stat& st2 = stats_[i % STATS];
      if (st2.seq != i) continue;
      char buf[200];
      std::snprintf(buf, sizeof buf, " #%lld[%zu B: %.0f, %.0f, %.0f, %.0f, %.0f]", (long long)st2.seq, st2.bytes, st2.wait_done, st2.memcpy_in, st2.wait_peer, st2.memcpy_out, st2.enqueue);
      s += buf;
    }
    return s;
  }

  // the helper's stage timings for the last `n` jobs, newest last: (seq, bytes, wait_for_card, memcpy_in, wait_for_peer, write_out, 0) in us
  std::vector<std::tuple<int64_t, size_t, double, double, double, double, double>> stats(int n) const {
    std::vector<std::tuple<int64_t, size_t, double, double, double, double, double>> out;
    int64_t tl = tail_.load();
    for (int64_t i = std::max<int64_t>(1, tl - n + 1); i <= tl; ++i) {
      const Stat& st = stats_[i % STATS];
      if (st.seq == i) out.emplace_back(st.seq, st.bytes, st.wait_done, st.memcpy_in, st.wait_peer, st.memcpy_out, st.enqueue);
    }
    return out;
  }

  void close() {
    if (closed_) return;
    closed_ = true;
    stop_.store(true, std::memory_order_release);
    if (helper_.joinable()) helper_.join();
    sycl::context& ctx = c10::xpu::get_device_context();
    for (int b = 0; b < 2; ++b) {
      if (mine_[b]) sycl::free(mine_[b], ctx);
      if (theirs_[b]) sycl::free(theirs_[b], ctx);
      if (mine_plain_[b]) sycl::free(mine_plain_[b], ctx);
      if (theirs_plain_[b]) sycl::free(theirs_plain_[b], ctx);
      if (theirs_dev_[b]) sycl::free(theirs_dev_[b], ctx);
    }
    if (status_) sycl::free(status_, ctx);
    if (marker_mine_) sycl::free(marker_mine_, ctx);
    if (marker_theirs_) sycl::free(marker_theirs_, ctx);
    for (auto& d : imported_.drivers) if (d.second && base_) d.second(d.first, base_);
    imported_.drivers.clear();
    if (base_) UnmapViewOfFile(base_);
    if (hmap_) CloseHandle(hmap_);
  }

 private:
  // Before the card is told to wait for the peer's half of s, make sure the peer has at least published s-1 (a plain
  // read of the shared flag). In lockstep this costs nothing; it keeps the Python thread at most one all_reduce ahead
  // of the slower card, so the GPU wait only ever covers one layer of peer work plus the hop, never a JIT compile or
  // a scheduler hiccup on the other side. A peer that is gone shows up here as a 60 s timeout.
  //
  // VLLM_XPU_EXCHANGE_AHEAD=K (2026-10-13, default 1 = the behaviour above): wait only for the peer's s-K, so the Python
  // thread can queue up to K all_reduces (and the layers between them) ahead of the cards. Correctness does not depend
  // on K: buffer reuse is ordered by the cards' in-order queues and the helper's in-order jobs (a card cannot export s+2
  // before its add s+1 has seen the peer's s+1, which the peer's helper publishes only after it has read my slot for s).
  // What K trades: a larger K hides the CPU's launch and Python time behind card work in two-card decode, but a peer
  // that stalls (a JIT compile, a scheduler hiccup) then leaves more queued card work spinning toward SPIN_LIMIT.
  void guard(int64_t seq) {
    if (seq <= ahead_) return;
    volatile int64_t* peer_flag = reinterpret_cast<volatile int64_t*>(base_ + FLAG_OFF[peer_]);
    if (!wait_word(peer_flag, seq - ahead_, stop_))
      throw std::runtime_error("xpu exchange: the peer never reached all_reduce #" + std::to_string(seq - ahead_) + " within " +
                               std::to_string((int)timeout_s()) + " s (peer gone or stalled; VLLM_XPU_EXCHANGE_TIMEOUT_S)");
  }

  // small path add kernel, one work-group: wait for the peer's tagged words (four per 32-byte load, scalar tail) and
  // add them in place
  template <int DT>
  static void add_kernel(sycl::queue& q, const uint32_t* x, uint32_t* y, const uint64_t* theirs, int64_t* status,
                         int64_t seq, uint32_t tag, size_t nwords) {
    const size_t nvec = nwords / 4;
    q.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
      size_t lid = it.get_local_id(0);
      for (size_t v = lid; v < nvec; v += WG) {
        u64x4 w = wait_tagged4(theirs + 4 * v, tag, seq, status);
        size_t i = 4 * v;
        y[i] = add_words<DT>(x[i], data_of(w[0]));
        y[i + 1] = add_words<DT>(x[i + 1], data_of(w[1]));
        y[i + 2] = add_words<DT>(x[i + 2], data_of(w[2]));
        y[i + 3] = add_words<DT>(x[i + 3], data_of(w[3]));
      }
      for (size_t i = 4 * nvec + lid; i < nwords; i += WG)
        y[i] = add_words<DT>(x[i], data_of(wait_tagged(theirs + i, tag, seq, status)));
    });
  }

  // small path, fused: export my tagged words, then wait for the peer's and add (one work-group, see run())
  template <int DT>
  static void fused_kernel(sycl::queue& q, const uint32_t* x, uint32_t* y, uint64_t* mine, const uint64_t* theirs,
                           int64_t* status, int64_t seq, uint32_t tag, size_t nwords) {
    const size_t nvec = nwords / 4;
    q.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
      size_t lid = it.get_local_id(0);
      {
        uncached_wptr<uint64_t> m{mine};
        for (size_t i = lid; i < nwords; i += WG) m[i] = tagged(x[i], tag);
      }
      for (size_t v = lid; v < nvec; v += WG) {
        u64x4 w = wait_tagged4(theirs + 4 * v, tag, seq, status);
        size_t i = 4 * v;
        y[i] = add_words<DT>(x[i], data_of(w[0]));
        y[i + 1] = add_words<DT>(x[i + 1], data_of(w[1]));
        y[i + 2] = add_words<DT>(x[i + 2], data_of(w[2]));
        y[i + 3] = add_words<DT>(x[i + 3], data_of(w[3]));
      }
      for (size_t i = 4 * nvec + lid; i < nwords; i += WG)
        y[i] = add_words<DT>(x[i], data_of(wait_tagged(theirs + i, tag, seq, status)));
    });
  }

  void check_status() const {
    volatile int64_t* s = reinterpret_cast<volatile int64_t*>(status_);
    if (s[0] != 0) throw std::runtime_error("xpu exchange: the card waited too long for the peer's half at all_reduce #" + std::to_string(s[0]) + " (peer gone or stalled); results after that point are invalid");
    if (s[8] != 0) throw std::runtime_error("xpu exchange: the helper thread gave up waiting at all_reduce #" + std::to_string(s[8]) + " (peer gone or stalled; VLLM_XPU_EXCHANGE_TIMEOUT_S)");
  }

  char* slot(int rank, int b) const { return base_ + HEADER + (size_t)(rank * 2 + b) * slot_bytes_; }

  void post(int64_t seq, size_t bytes) {
    int64_t h = head_.load(std::memory_order_relaxed);
    while (h - tail_.load(std::memory_order_acquire) >= RING) _mm_pause();
    ring_[h % RING] = Job{seq, bytes};
    head_.store(h + 1, std::memory_order_release);
  }

  // spin on a host word until it reaches `want`; false on timeout. For a word the *card* writes, invalidate this core's
  // cached copy of the line before every read: the card's writes into host memory do not invalidate CPU cache lines.
  static bool wait_word(volatile int64_t* w, int64_t want, const std::atomic<bool>& stop, bool card_written = false) {
    auto t0 = std::chrono::steady_clock::now();
    const double limit = timeout_s();
    uint32_t n = 0;
    for (;;) {
      if (card_written) { _mm_clflush(const_cast<const int64_t*>(w)); _mm_mfence(); }
      if (*w >= want) return true;
      if (stop.load(std::memory_order_relaxed)) return false;
      _mm_pause();
      if ((++n & 0xfff) == 0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > limit) return false;
    }
  }

  // Read the card's tagged export of job `seq` into `dst` (plain data words). Every word must carry the tag; a word that
  // does not yet is re-read after invalidating its cache line (the card's write may still be on its way, and the CPU's
  // cached copy of the line is not invalidated by it). False on timeout.
  bool collect_export(uint32_t* dst, const uint64_t* src, size_t nwords, uint32_t tag) {
    auto t0 = std::chrono::steady_clock::now();
    const double limit = timeout_s();
    invalidate_lines(src, nwords * sizeof(uint64_t));
    for (size_t i = 0; i < nwords; ++i) {
      uint64_t v = reinterpret_cast<const volatile uint64_t*>(src)[i];
      uint32_t n = 0;
      while (tag_of(v) != tag) {
        _mm_clflush(const_cast<uint64_t*>(src) + i);
        _mm_mfence();
        _mm_pause();
        v = reinterpret_cast<const volatile uint64_t*>(src)[i];
        if ((++n & 0xfff) == 0) {
          if (stop_.load(std::memory_order_relaxed)) return false;
          if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > limit) return false;
        }
      }
      dst[i] = data_of(v);
    }
    return true;
  }

  void helper_main() {
    volatile int64_t* my_flag = reinterpret_cast<volatile int64_t*>(base_ + FLAG_OFF[rank_]);
    volatile int64_t* peer_flag = reinterpret_cast<volatile int64_t*>(base_ + FLAG_OFF[peer_]);
    volatile int64_t* status = reinterpret_cast<volatile int64_t*>(status_);
    while (!stop_.load(std::memory_order_acquire)) {
      int64_t t = tail_.load(std::memory_order_acquire);
      if (t == head_.load(std::memory_order_acquire)) { _mm_pause(); continue; }
      Job job = ring_[t % RING];
      int b = (int)(job.seq & 1);
      uint32_t tag = (uint32_t)job.seq;
      size_t nwords = job.bytes / 4;
      using clk = std::chrono::steady_clock;
      auto us = [](clk::time_point a, clk::time_point b2) { return std::chrono::duration<double, std::micro>(b2 - a).count(); };
      const bool small = job.bytes <= SMALL;
      auto t0 = clk::now();
      // 1. the card's half into my shared slot
      if (small) {
        // checked word by word
        if (!collect_export(reinterpret_cast<uint32_t*>(slot(rank_, b)), mine_[b], nwords, tag)) {
          if (!stop_) status[8] = job.seq; tail_.store(t + 1); continue;
        }
      } else {
        // the marker says the copy engine has finished; drop this core's stale lines, then copy
        if (!wait_word(reinterpret_cast<volatile int64_t*>(marker_mine_) + b * 8, job.seq, stop_, /*card_written=*/true)) {
          if (!stop_) status[8] = job.seq; tail_.store(t + 1); continue;
        }
        if (!direct_) {
          invalidate_lines(mine_plain_[b], job.bytes);
          _mm_mfence();
          std::memcpy(slot(rank_, b), mine_plain_[b], job.bytes);
        }
      }
      auto t1 = clk::now();
      MemoryBarrier();
      *my_flag = job.seq;
      auto t2 = clk::now();
      // 2. the peer's half
      if (!wait_word(peer_flag, job.seq, stop_)) { if (!stop_) status[8] = job.seq; tail_.store(t + 1); continue; }
      auto t3 = clk::now();
      MemoryBarrier();
      if (small) {
        const uint32_t* peer = reinterpret_cast<const uint32_t*>(slot(peer_, b));
        uint64_t* out = theirs_[b];
        for (size_t i = 0; i < nwords; ++i) out[i] = tagged(peer[i], tag);
        // push the tagged words out of this core's cache so the card's cache-bypassing loads find them in memory
        invalidate_lines(out, nwords * sizeof(uint64_t));
        _mm_mfence();
      } else {
        if (!direct_) {
          std::memcpy(theirs_plain_[b], slot(peer_, b), job.bytes);
          invalidate_lines(theirs_plain_[b], job.bytes);
        }
        _mm_mfence();
        reinterpret_cast<volatile int64_t*>(marker_theirs_)[b * 8] = job.seq;   // the card's one-lane kernel polls this
        _mm_mfence();
      }
      auto t4 = clk::now();
      stats_[job.seq % STATS] = Stat{job.seq, job.bytes, us(t0, t1), us(t1, t2), us(t2, t3), us(t3, t4), 0.0};
      tail_.store(t + 1, std::memory_order_release);
    }
  }

  int rank_, peer_;
  size_t slot_bytes_;
  int device_index_;
  HANDLE hmap_ = NULL;
  char* base_ = nullptr;
  size_t total_ = 0;
  bool direct_ = false;          // the section is imported for the copy engine: the big path skips the helper's memcpy
  bool fused_ = false;           // small path in one kernel (export, wait, add); VLLM_XPU_EXCHANGE_FUSED=1
  int64_t ahead_ = 1;            // all_reduces the Python thread may run ahead of the peer; VLLM_XPU_EXCHANGE_AHEAD
  Imported imported_;
  uint64_t* mine_[2] = {};       // host USM, tagged words (small path): my half, written by the export kernel, read by the helper
  uint64_t* theirs_[2] = {};     // host USM, tagged words (small path): the peer's half, written by the helper, read by the add kernel
  char* mine_plain_[2] = {};     // host USM (big path): my half, written by the copy engine, read by the helper
  char* theirs_plain_[2] = {};   // host USM (big path): the peer's half, written by the helper, read by the copy engine
  char* theirs_dev_[2] = {};     // card memory (big path): the peer's half, copied in, read by the add kernel
  int64_t* marker_mine_ = nullptr;   // host USM, 2 lines (big path): one-lane kernel -> helper, "the copy has finished"
  int64_t* marker_theirs_ = nullptr; // host USM, 2 lines (big path): helper -> one-lane kernel, "the peer's half is in place"
  int64_t* status_ = nullptr;    // host USM, 2 lines: [0] card gave up at seq, [8] helper gave up at seq
  int64_t seq_ = 0;
  int64_t n_small_ = 0, n_big_ = 0;
  Stat stats_[STATS] = {};
  std::thread helper_;
  Job ring_[RING];
  std::atomic<int64_t> head_{0}, tail_{0};
  std::atomic<bool> stop_{false};
  bool closed_ = false;
};

}  // namespace

PYBIND11_MODULE(_xpu_exchange, m) {
  m.doc() = "two-card all_reduce for vLLM on Windows XPU: tagged words through host memory, GPU wait and add";
  pybind11::class_<Exchange>(m, "Exchange")
      .def(pybind11::init<int, int, const std::string&, size_t, int>(), pybind11::arg("rank"), pybind11::arg("world_size"),
           pybind11::arg("name"), pybind11::arg("slot_bytes"), pybind11::arg("device_index"))
      .def("all_reduce_", &Exchange::all_reduce_, pybind11::arg("tensor"), "all_reduce on the current XPU stream into a new tensor; returns at once")
      .def("try_all_reduce_", &Exchange::try_all_reduce_, pybind11::arg("tensor"), "as all_reduce_, but returns None (and does nothing) for a tensor the exchange cannot take")
      .def("status", &Exchange::status, "(gpu_timeout_seq, helper_timeout_seq, sequence); zeros mean healthy")
      .def("debug", &Exchange::debug, "one line with every flag and the last jobs, for the log when something fails")
      .def("stats", &Exchange::stats, pybind11::arg("n") = 16, "helper stage timings (us) for the last n jobs")
      .def("direct", &Exchange::direct, "True when the shared section is imported and the big path uses it with the copy engine")
      .def("close", &Exchange::close);
}
