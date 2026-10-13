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
// shared header: one cache line per word. SMALL/BIG = per-rank published sequence of each channel; HOSTCNT = per-rank
// count of host calls (the eager guard)
constexpr size_t FLAG_SMALL_OFF[2] = {0, 64};
constexpr size_t FLAG_BIG_OFF[2] = {128, 192};
constexpr size_t HOSTCNT_OFF[2] = {256, 320};
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
constexpr int HRING = 1024;                     // small-path job headers, card -> helper

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

// GRAPH-SAFE LAYOUT (2026-10-13). Two channels, so the decode-size exchanges can be recorded into an XPU graph and replayed:
//   small (<= 64 KB, decode): NOTHING per call on the host. The sequence number lives in card memory (dseq_): the export
//     kernel reads s = dseq + 1, writes its tagged words into mine[s & 1] and a header word (bytes, tag s) into hdr_[s %
//     HRING]; the add kernel reads the same s, waits for the peer's words tagged s in theirs[s & 1], adds, and its lane 0
//     stores dseq = s. A replayed graph therefore gets a fresh s every time. The helper finds each job by watching hdr_
//     (no host post), relays through its own small slots and SMALL flags.
//   big (> 64 KB, prefill chunks): host-driven as before (host counter bseq_, posts, markers, BIG flags, direct copy-engine
//     slots), and refused while the queue is recording a graph: its slot parity and markers are host values that a
//     replay could not refresh. vLLM's capture sizes are capped so decode-size batches are the only ones captured.
// The helper is one non-blocking loop that advances both channels, so neither can stall the other. A host-level guard
// (each side's count of host calls, in the shared header) keeps eager callers in step; replays never touch the host.
class Exchange {
 public:
  Exchange(int rank, int world_size, const std::string& name, size_t slot_bytes, int device_index)
      : rank_(rank), peer_(1 - rank), slot_bytes_(slot_bytes), device_index_(device_index) {
    if (world_size != 2) throw std::invalid_argument("xpu exchange is written for two cards");
    const size_t total = HEADER + 4 * slot_bytes + 4 * SMALL;
    if (rank == 0) {
      hmap_ = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, (DWORD)(total >> 32), (DWORD)(total & 0xffffffffu), name.c_str());
    } else {
      for (int i = 0; i < 600 && !hmap_; ++i) { hmap_ = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, name.c_str()); if (!hmap_) Sleep(50); }
    }
    if (!hmap_) throw std::runtime_error("xpu exchange: shared mapping '" + name + "' failed, error " + std::to_string(GetLastError()));
    base_ = reinterpret_cast<char*>(MapViewOfFile(hmap_, FILE_MAP_ALL_ACCESS, 0, 0, total));
    if (!base_) throw std::runtime_error("xpu exchange: MapViewOfFile failed, error " + std::to_string(GetLastError()));
    if (rank == 0) std::memset(base_, 0, HEADER);   // flags and host counts; the slots need no zeroing
    total_ = total;
    direct_ = import_for_copy_engine(base_, total, imported_);
    { const char* a = std::getenv("VLLM_XPU_EXCHANGE_AHEAD"); int k = a ? std::atoi(a) : 1; ahead_ = k < 1 ? 1 : (k > 64 ? 64 : k); }
    sycl::context& ctx = c10::xpu::get_device_context();
    sycl::device& dev = c10::xpu::get_raw_device(device_index);
    for (int b = 0; b < 2; ++b) {
      mine_[b] = static_cast<uint64_t*>(sycl::malloc_host(2 * SMALL, ctx));
      theirs_[b] = static_cast<uint64_t*>(sycl::malloc_host(2 * SMALL, ctx));
      if (!direct_) {
        mine_plain_[b] = static_cast<char*>(sycl::malloc_host(slot_bytes, ctx));
        theirs_plain_[b] = static_cast<char*>(sycl::malloc_host(slot_bytes, ctx));
      }
      theirs_dev_[b] = static_cast<char*>(sycl::malloc_device(slot_bytes, dev, ctx));
    }
    status_ = static_cast<int64_t*>(sycl::malloc_host(2 * 64, ctx));
    marker_mine_ = static_cast<int64_t*>(sycl::malloc_host(2 * 64, ctx));
    marker_theirs_ = static_cast<int64_t*>(sycl::malloc_host(2 * 64, ctx));
    hdr_ = static_cast<uint64_t*>(sycl::malloc_host(HRING * sizeof(uint64_t), ctx));
    dseq_ = static_cast<int64_t*>(sycl::malloc_device(64, dev, ctx));
    for (int b = 0; b < 2; ++b)
      if (!mine_[b] || !theirs_[b] || (!direct_ && (!mine_plain_[b] || !theirs_plain_[b])) || !theirs_dev_[b])
        throw std::runtime_error("xpu exchange: USM allocation failed");
    if (!status_ || !marker_mine_ || !marker_theirs_ || !hdr_ || !dseq_) throw std::runtime_error("xpu exchange: USM allocation failed");
    for (int b = 0; b < 2; ++b) { std::memset(mine_[b], 0, 2 * SMALL); std::memset(theirs_[b], 0, 2 * SMALL); }
    std::memset(status_, 0, 2 * 64); std::memset(marker_mine_, 0, 2 * 64); std::memset(marker_theirs_, 0, 2 * 64);
    std::memset(hdr_, 0, HRING * sizeof(uint64_t));
    c10::xpu::getCurrentXPUStream(device_index).queue().memset(dseq_, 0, 64).wait();
    helper_ = std::thread(&Exchange::helper_main, this);
  }

  ~Exchange() { close(); }

  // The hot path: no tensor when this one is not for the exchange (dtype, size, layout) so the caller can fall back,
  // without a second round of Python checks per layer; raises if an earlier exchange failed.
  // Out of place on purpose: vLLM registers all_reduce as a functional custom op (the fake returns empty_like), so under
  // torch.compile Inductor reuses the input's buffer right after the call; an in-place result was overwritten by the next
  // kernel and the compiled two-card run produced garbage (2026-10-11 12:00).
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
    const bool recording = q.ext_oneapi_get_state() == oex::queue_state::recording;
    uint32_t* x = static_cast<uint32_t*>(t.data_ptr());
    uint32_t* y = static_cast<uint32_t*>(out.data_ptr());
    const size_t nwords = bytes / 4;
    if (bytes <= SMALL) {
      // Small path: everything the call needs is read from card memory at run time, so the two kernels can be recorded
      // once and replayed. Export: tagged words into mine[s & 1] and the job header for the helper.
      ++n_small_;
      if (recording) ++n_recorded_;
      uint64_t* m0 = mine_[0];
      uint64_t* m1 = mine_[1];
      uint64_t* hdr = hdr_;
      const int64_t* dseq_r = dseq_;
      const uint32_t nb = (uint32_t)bytes;
      q.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t s = dseq_r[0] + 1;
        const uint32_t tag = (uint32_t)s;
        uncached_wptr<uint64_t> m{(s & 1) ? m1 : m0};
        const size_t lid = it.get_local_id(0);
        for (size_t i = lid; i < nwords; i += WG) m[i] = tagged(x[i], tag);
        if (lid == 0) { uncached_wptr<uint64_t> h{hdr + (size_t)(s % HRING)}; h[0] = tagged(nb, tag); }
      });
      guard();
      const uint64_t* t0 = theirs_[0];
      const uint64_t* t1 = theirs_[1];
      int64_t* dseq = dseq_;
      int64_t* status = status_;
      switch (dt) {
        case F32: add_kernel<F32>(q, x, y, t0, t1, dseq, status, nwords); break;
        case BF16: add_kernel<BF16>(q, x, y, t0, t1, dseq, status, nwords); break;
        default: add_kernel<F16>(q, x, y, t0, t1, dseq, status, nwords); break;
      }
      return out;
    }
    // Big path (prefill chunks, MBs): host-driven, so never inside a graph.
    TORCH_CHECK(!recording, "xpu exchange: an all_reduce of ", bytes, " B (over 64 KB) inside an XPU graph capture: the "
                "large path is host-driven and a replay would reuse stale slots. Cap the cudagraph capture sizes so only "
                "decode-size batches are captured.");
    ++n_big_;
    const int64_t seq = ++bseq_;
    const int b = (int)(seq & 1);
    // the copy engine moves the bytes straight into my slot of the imported section (direct_) or into pinned staging;
    // a one-lane kernel, started by the in-order queue only after the copy, stores the marker the helper waits for
    q.memcpy(direct_ ? slot(rank_, b) : mine_plain_[b], x, bytes);
    int64_t* marker = marker_mine_ + b * 8;
    q.single_task([=]() { uncached_wptr<int64_t> m{marker}; m[0] = seq; });
    post(seq, bytes);
    guard();
    int64_t* status = status_;
    const int64_t* marker_t = marker_theirs_ + b * 8;
    q.single_task([=]() {
      uncached_rptr<const int64_t> r{marker_t};
      uint32_t polls = 0;
      for (;;) {
        int64_t v = r[0];
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
        if (v >= seq) break;
        if (++polls > SPIN_LIMIT) { uncached_wptr<int64_t> s{status}; if (s[0] == 0) s[0] = -seq; break; }
      }
    });
    // straight out of the peer's slot (direct_). Reuse is safe: my copy into slot b at seq+2 runs after my poll at seq+1,
    // which needs the peer's BIG flag seq+1, published only after its card has read my slot b at seq.
    q.memcpy(theirs_dev_[b], direct_ ? slot(peer_, b) : theirs_plain_[b], bytes);
    const uint32_t* theirs = reinterpret_cast<const uint32_t*>(theirs_dev_[b]);
    switch (dt) {
      case F32: q.parallel_for(sycl::range<1>(nwords), [=](sycl::id<1> i) { y[i] = add_words<F32>(x[i], theirs[i]); }); break;
      case BF16: q.parallel_for(sycl::range<1>(nwords), [=](sycl::id<1> i) { y[i] = add_words<BF16>(x[i], theirs[i]); }); break;
      default: q.parallel_for(sycl::range<1>(nwords), [=](sycl::id<1> i) { y[i] = add_words<F16>(x[i], theirs[i]); }); break;
    }
    return out;
  }

  // small path add, one work-group: s from card memory, wait for the peer's words tagged s (four per 32-byte load, scalar
  // tail), add into the fresh output, then lane 0 advances dseq (after a barrier: every lane has read it by then)
  template <int DT>
  static void add_kernel(sycl::queue& q, const uint32_t* x, uint32_t* y, const uint64_t* t0, const uint64_t* t1,
                         int64_t* dseq, int64_t* status, size_t nwords) {
    const size_t nvec = nwords / 4;
    q.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
      const int64_t s = dseq[0] + 1;
      const uint32_t tag = (uint32_t)s;
      const uint64_t* theirs = (s & 1) ? t1 : t0;
      const size_t lid = it.get_local_id(0);
      for (size_t v = lid; v < nvec; v += WG) {
        u64x4 w = wait_tagged4(theirs + 4 * v, tag, s, status);
        size_t i = 4 * v;
        y[i] = add_words<DT>(x[i], data_of(w[0]));
        y[i + 1] = add_words<DT>(x[i + 1], data_of(w[1]));
        y[i + 2] = add_words<DT>(x[i + 2], data_of(w[2]));
        y[i + 3] = add_words<DT>(x[i + 3], data_of(w[3]));
      }
      for (size_t i = 4 * nvec + lid; i < nwords; i += WG)
        y[i] = add_words<DT>(x[i], data_of(wait_tagged(theirs + i, tag, s, status)));
      sycl::group_barrier(it.get_group());
      if (lid == 0) dseq[0] = s;
    });
  }

 public:
  bool direct() const { return direct_; }
  // (card_timeout_seq, helper_timeout_seq, host_calls): zeros mean healthy; a negative card value is a big-path sequence
  std::tuple<int64_t, int64_t, int64_t> status() const {
    return {reinterpret_cast<volatile int64_t*>(status_)[0], reinterpret_cast<volatile int64_t*>(status_)[8], host_calls_};
  }

  // calls made from the host so far, small ones while a graph was being recorded, and big ones
  std::tuple<int64_t, int64_t, int64_t> counts() const { return {n_small_, n_recorded_, n_big_}; }

  // Everything the two sides have said to each other, for the log when an all_reduce fails.
  std::string debug() {
    volatile int64_t* st = reinterpret_cast<volatile int64_t*>(status_);
    auto word = [&](size_t off) { return std::to_string(*reinterpret_cast<volatile int64_t*>(base_ + off)); };
    std::string s = "rank " + std::to_string(rank_) + ": host calls " + std::to_string(host_calls_) +
        " (small " + std::to_string(n_small_) + ", recorded " + std::to_string(n_recorded_) + ", big " + std::to_string(n_big_) +
        "), card_timeout_at " + std::to_string(st[0]) + ", helper_timeout_at " + std::to_string(st[8]) +
        ", small flags mine/peer " + word(FLAG_SMALL_OFF[rank_]) + "/" + word(FLAG_SMALL_OFF[peer_]) +
        ", big flags mine/peer " + word(FLAG_BIG_OFF[rank_]) + "/" + word(FLAG_BIG_OFF[peer_]) +
        ", host counts mine/peer " + word(HOSTCNT_OFF[rank_]) + "/" + word(HOSTCNT_OFF[peer_]) +
        ", helper next small " + std::to_string(next_small_.load()) + ", big head/tail " + std::to_string(head_.load()) + "/" +
        std::to_string(tail_.load()) + ", direct " + (direct_ ? "yes" : "no");
    return s;
  }

  // the helper's stage timings for the last `n` jobs of either channel, oldest first:
  // (seq, bytes, wait_for_card, publish, wait_for_peer, write_out, 0) in us; big jobs carry a negative seq
  std::vector<std::tuple<int64_t, size_t, double, double, double, double, double>> stats(int n) const {
    std::vector<std::tuple<int64_t, size_t, double, double, double, double, double>> out;
    const int64_t c = stat_count_.load();
    for (int64_t i = std::max<int64_t>(0, c - n); i < c; ++i) {
      const Stat& st = stats_[i % STATS];
      out.emplace_back(st.seq, st.bytes, st.wait_done, st.memcpy_in, st.wait_peer, st.memcpy_out, st.enqueue);
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
    if (hdr_) sycl::free(hdr_, ctx);
    if (dseq_) sycl::free(dseq_, ctx);
    for (auto& d : imported_.drivers) if (d.second && base_) d.second(d.first, base_);
    imported_.drivers.clear();
    if (base_) UnmapViewOfFile(base_);
    if (hmap_) CloseHandle(hmap_);
  }

 private:
  // Keep the two hosts in step: publish my count of host calls, then wait until the peer has made at least as many
  // (minus VLLM_XPU_EXCHANGE_AHEAD - 1). The export is already queued, so both sides queue their export, then meet here;
  // the card's add therefore waits only for the peer's CARD, never for a peer host that is still compiling. Graph
  // replays do not come through here at all.
  void guard() {
    const int64_t mine = ++host_calls_;
    volatile int64_t* my = reinterpret_cast<volatile int64_t*>(base_ + HOSTCNT_OFF[rank_]);
    volatile int64_t* pe = reinterpret_cast<volatile int64_t*>(base_ + HOSTCNT_OFF[peer_]);
    *my = mine;
    const int64_t want = mine - (ahead_ - 1);
    if (want <= 0) return;
    if (!wait_word(pe, want, stop_))
      throw std::runtime_error("xpu exchange: the peer never made host call #" + std::to_string(want) + " within " +
                               std::to_string((int)timeout_s()) + " s (peer gone or stalled; VLLM_XPU_EXCHANGE_TIMEOUT_S)");
  }

  void check_status() const {
    volatile int64_t* s = reinterpret_cast<volatile int64_t*>(status_);
    if (s[0] != 0) throw std::runtime_error("xpu exchange: the card waited too long for the peer's half at all_reduce #" + std::to_string(s[0]) + " (negative = big path; peer gone or stalled); results after that point are invalid");
    if (s[8] != 0) throw std::runtime_error("xpu exchange: the helper thread gave up waiting at all_reduce #" + std::to_string(s[8]) + " (peer gone or stalled; VLLM_XPU_EXCHANGE_TIMEOUT_S)");
  }

  char* slot(int rank, int b) const { return base_ + HEADER + (size_t)(rank * 2 + b) * slot_bytes_; }
  char* small_slot(int rank, int b) const { return base_ + HEADER + 4 * slot_bytes_ + (size_t)(rank * 2 + b) * SMALL; }

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

  // Read the card's tagged export into `dst` (plain data words). Every word must carry the tag; a word that does not yet
  // is re-read after invalidating its cache line. False on timeout.
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

  void record(int64_t seq, size_t bytes, double a, double b2, double c, double d) {
    const int64_t i = stat_count_.load(std::memory_order_relaxed);
    stats_[i % STATS] = Stat{seq, bytes, a, b2, c, d, 0.0};
    stat_count_.store(i + 1, std::memory_order_release);
  }

  // One loop, two channels, never blocking on either: a stalled channel must not hold up the other, because the card runs
  // them interleaved in program order.
  void helper_main() {
    using clk = std::chrono::steady_clock;
    auto us = [](clk::time_point a, clk::time_point b2) { return std::chrono::duration<double, std::micro>(b2 - a).count(); };
    volatile int64_t* fs_my = reinterpret_cast<volatile int64_t*>(base_ + FLAG_SMALL_OFF[rank_]);
    volatile int64_t* fs_peer = reinterpret_cast<volatile int64_t*>(base_ + FLAG_SMALL_OFF[peer_]);
    volatile int64_t* fb_my = reinterpret_cast<volatile int64_t*>(base_ + FLAG_BIG_OFF[rank_]);
    volatile int64_t* fb_peer = reinterpret_cast<volatile int64_t*>(base_ + FLAG_BIG_OFF[peer_]);
    volatile int64_t* status = reinterpret_cast<volatile int64_t*>(status_);
    const double limit = timeout_s();
    int64_t sn = 1;          // the next small sequence the card will export
    int sst = 0;             // 0: waiting for its header, 1: published, waiting for the peer
    size_t sbytes = 0;
    clk::time_point s0, s1, s2;
    int bst = 0;             // big channel: 0: waiting for a posted job's marker, 1: published, waiting for the peer
    Job bj{0, 0};
    clk::time_point b0, b1;
    clk::time_point last = clk::now();
    bool failed = false;
    while (!stop_.load(std::memory_order_acquire)) {
      bool moved = false;
      // ---- small channel
      if (sst == 0) {
        volatile uint64_t* h = reinterpret_cast<volatile uint64_t*>(hdr_ + (size_t)(sn % HRING));
        _mm_clflush(const_cast<const uint64_t*>(h));
        _mm_mfence();
        const uint64_t hv = *h;
        if (tag_of(hv) == (uint32_t)sn) {
          s0 = clk::now();
          sbytes = data_of(hv);
          const int b = (int)(sn & 1);
          if (!collect_export(reinterpret_cast<uint32_t*>(small_slot(rank_, b)), mine_[b], sbytes / 4, (uint32_t)sn)) {
            if (!stop_) status[8] = sn;
            failed = true;
          } else {
            MemoryBarrier();
            *fs_my = sn;
            s1 = clk::now();
            sst = 1;
            moved = true;
          }
        }
      } else if (*fs_peer >= sn) {
        s2 = clk::now();
        MemoryBarrier();
        const int b = (int)(sn & 1);
        const uint32_t* peer = reinterpret_cast<const uint32_t*>(small_slot(peer_, b));
        uint64_t* out = theirs_[b];
        const size_t nwords = sbytes / 4;
        const uint32_t tag = (uint32_t)sn;
        for (size_t i = 0; i < nwords; ++i) out[i] = tagged(peer[i], tag);
        // push the tagged words out of this core's cache so the card's cache-bypassing loads find them in memory
        invalidate_lines(out, nwords * sizeof(uint64_t));
        _mm_mfence();
        record(sn, sbytes, 0.0, us(s0, s1), us(s1, s2), us(s2, clk::now()));
        next_small_.store(sn + 1, std::memory_order_relaxed);
        ++sn;
        sst = 0;
        moved = true;
      }
      // ---- big channel
      if (bst == 0) {
        const int64_t t = tail_.load(std::memory_order_acquire);
        if (t < head_.load(std::memory_order_acquire)) {
          bj = ring_[t % RING];
          volatile int64_t* m = reinterpret_cast<volatile int64_t*>(marker_mine_) + (bj.seq & 1) * 8;
          _mm_clflush(const_cast<const int64_t*>(m));
          _mm_mfence();
          if (*m >= bj.seq) {
            b0 = clk::now();
            const int b = (int)(bj.seq & 1);
            if (!direct_) {
              invalidate_lines(mine_plain_[b], bj.bytes);
              _mm_mfence();
              std::memcpy(slot(rank_, b), mine_plain_[b], bj.bytes);
            }
            MemoryBarrier();
            *fb_my = bj.seq;
            b1 = clk::now();
            bst = 1;
            moved = true;
          }
        }
      } else if (*fb_peer >= bj.seq) {
        auto b2 = clk::now();
        MemoryBarrier();
        const int b = (int)(bj.seq & 1);
        if (!direct_) {
          std::memcpy(theirs_plain_[b], slot(peer_, b), bj.bytes);
          invalidate_lines(theirs_plain_[b], bj.bytes);
        }
        _mm_mfence();
        reinterpret_cast<volatile int64_t*>(marker_theirs_)[b * 8] = bj.seq;   // the card's one-lane kernel polls this
        _mm_mfence();
        record(-bj.seq, bj.bytes, 0.0, us(b0, b1), us(b1, b2), us(b2, clk::now()));
        tail_.store(tail_.load() + 1, std::memory_order_release);
        bst = 0;
        moved = true;
      }
      if (failed) break;
      if (moved) { last = clk::now(); continue; }
      _mm_pause();
      // a channel that has published and waits for the peer, or a posted big job whose card never signals, for longer
      // than the timeout means the peer (or this card) is gone
      const bool waiting = sst == 1 || bst == 1 || (bst == 0 && tail_.load() < head_.load());
      if (waiting && std::chrono::duration<double>(clk::now() - last).count() > limit) {
        if (status[8] == 0) status[8] = sst == 1 ? sn : -bj.seq;
        break;
      }
    }
  }

  int rank_, peer_;
  size_t slot_bytes_;
  int device_index_;
  HANDLE hmap_ = NULL;
  char* base_ = nullptr;
  size_t total_ = 0;
  bool direct_ = false;          // the section is imported for the copy engine: the big path skips the helper's memcpy
  Imported imported_;
  int64_t ahead_ = 1;            // host calls this side may make ahead of the peer's; VLLM_XPU_EXCHANGE_AHEAD
  uint64_t* mine_[2] = {};       // host USM, tagged words (small path): my half, written by the export kernel, read by the helper
  uint64_t* theirs_[2] = {};     // host USM, tagged words (small path): the peer's half, written by the helper, read by the add kernel
  char* mine_plain_[2] = {};     // host USM (big path, not direct): my half, written by the copy engine, read by the helper
  char* theirs_plain_[2] = {};   // host USM (big path, not direct): the peer's half, written by the helper, read by the copy engine
  char* theirs_dev_[2] = {};     // card memory (big path): the peer's half, copied in, read by the add kernel
  int64_t* marker_mine_ = nullptr;   // host USM, 2 lines (big path): one-lane kernel -> helper, "the copy has finished"
  int64_t* marker_theirs_ = nullptr; // host USM, 2 lines (big path): helper -> one-lane kernel, "the peer's half is in place"
  int64_t* status_ = nullptr;    // host USM, 2 lines: [0] card gave up at seq, [8] helper gave up at seq
  uint64_t* hdr_ = nullptr;      // host USM, HRING words (small path): card -> helper job headers (bytes, tag s)
  int64_t* dseq_ = nullptr;      // card memory (small path): the last small sequence completed on this card
  int64_t bseq_ = 0;             // host: big-path sequence
  int64_t host_calls_ = 0;       // host: calls through run(), small and big, for the guard
  int64_t n_small_ = 0, n_big_ = 0, n_recorded_ = 0;
  Stat stats_[STATS] = {};
  std::atomic<int64_t> stat_count_{0};
  std::atomic<int64_t> next_small_{1};
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
      .def("status", &Exchange::status, "(card_timeout_seq, helper_timeout_seq, host_calls); zeros mean healthy")
      .def("debug", &Exchange::debug, "one line with every flag and the last jobs, for the log when something fails")
      .def("stats", &Exchange::stats, pybind11::arg("n") = 16, "helper stage timings (us) for the last n jobs")
      .def("direct", &Exchange::direct, "True when the shared section is imported and the big path uses it with the copy engine")
      .def("counts", &Exchange::counts, "(small host calls, small calls made while recording a graph, big host calls)")
      .def("close", &Exchange::close);
}
