// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// xpu_exchange1p.cpp - the two-card all_reduce and all_gather for a ONE-PROCESS two-card vLLM on Windows XPU (the
// one-process executor, v1/executor/oneproc_executor.py). Built by kernels\build-exchange1p.input.cmd into
// _xpu_exchange1p.pyd.
//
// With both cards in one process (one SYCL context), a kernel on either card can read and write the same pinned host
// memory, so no helper thread and no shared section are needed (the two-process exchange, xpu_exchange.cpp, relays every
// decode all_reduce through a CPU helper per process: 28 us). probe/one-process-allreduce (2026-10-13): the decode
// all_reduce of the 7B (1,792 tagged words) as one fused kernel per card takes 7.0 us launched, 4.4 us persistent, when
// the words are spread over 16-32 work-groups; with one work-group it took 56 us and wider (32/64-byte) stores were 4-15x
// slower, so the stores stay 8-byte tagged words and the work is spread over groups.
//
// Small path (<= 64 KB, decode), one kernel per card per call, graph-safe (nothing host-side per call):
//   every group reads s = dseq[0] + 1 (card memory), writes its slice of my words tagged s into the peer's inbox
//   [s & 1] (pinned host), waits for the peer's slice tagged s in my inbox [s & 1], adds (or places, for a gather) into a
//   fresh output; the last group to finish (a device-scope counter) stores dseq = s.
//   Parity reuse is safe: a card reaches call s+2 only after call s+1 completed, which needed every word of the peer's
//   call s+1, which the peer writes only after its call s has completed (in-order queues), i.e. after it read inbox [s & 1].
// Big path (> 64 KB, prefill chunks, the logits gather): the copy engine moves my bytes into my pinned slot [b], a one-lane
// kernel stores the marker, a one-lane kernel polls the peer's marker, the copy engine brings the peer's bytes into card
// memory, a plain kernel adds. Host-driven sequence (per card), so it is refused inside a graph capture.
// Both cards must make the same calls in the same order (SPMD), exactly as with the two-process exchange.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <immintrin.h>
#include <atomic>
#include <chrono>
#include <torch/extension.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

namespace {

constexpr int WG = 512;
constexpr int MAX_GROUPS = 32;                 // probe: 1,792 words 4.4 us at 32 groups; 64 groups misbehaved at 1,792
constexpr size_t WORDS_PER_GROUP = 64;
constexpr size_t SMALL = 64u << 10;            // data bytes per call on the small path
constexpr uint32_t SPIN_LIMIT = 6000000;       // ~1 s of polls, under the 2 s Windows TDR

namespace iex = sycl::ext::intel::experimental;
namespace oex = sycl::ext::oneapi::experimental;
using uncached_wprops = decltype(oex::properties(
    iex::write_hint<iex::cache_control<iex::cache_mode::uncached, oex::cache_level::L1, oex::cache_level::L3>>));
template <class T> using uncached_wptr = oex::annotated_ptr<T, uncached_wprops>;
using uncached_rprops = decltype(oex::properties(
    iex::read_hint<iex::cache_control<iex::cache_mode::uncached, oex::cache_level::L1, oex::cache_level::L2, oex::cache_level::L3>>));
template <class T> using uncached_rptr = oex::annotated_ptr<T, uncached_rprops>;

enum DType : int { F32 = 0, BF16 = 1, F16 = 2, RAW = 3 };   // RAW: gather, no arithmetic
enum Mode : int { ADD = 0, GATHER = 1 };
// FUSED: one call does both halves (graph capture, where both cards' replays are queued back to back). In eager, a card's
// waiting kernel must not be queued before the other card's export is: the other rank's HOST may still be seconds away
// from that call (first-use kernel compiles), and a waiting kernel gives up after ~1 s (the 2 s Windows GPU timeout).
// So eager calls are split: EXPORT (my half out), hand the baton over, FINISH (wait for the other half, combine).
// 2026-10-13 17:08: the first prefill all_reduce of vLLM's memory profiling timed out exactly that way.
enum Phase : int { FUSED = 0, EXPORT = 1, FINISH = 2 };

inline uint64_t tagged(uint32_t data, uint32_t tag) { return (uint64_t)data | ((uint64_t)tag << 32); }
inline uint32_t data_of(uint64_t w) { return (uint32_t)w; }
inline uint32_t tag_of(uint64_t w) { return (uint32_t)(w >> 32); }

// as xpu_exchange.cpp: float math, round to nearest even, so both cards hold identical bytes (a + b == b + a)
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

inline uint64_t wait_tagged(const uint64_t* p, uint32_t tag, int64_t seq, int64_t* status) {
  uncached_rptr<const uint64_t> r{p};
  uint32_t polls = 0;
  for (;;) {
    uint64_t v = r[0];
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    if (tag_of(v) == tag) return v;
    if (++polls > SPIN_LIMIT) { uncached_wptr<int64_t> s{status}; if (s[0] == 0) s[0] = seq; return v; }
  }
}

class Pair {
 public:
  explicit Pair(size_t slot_bytes) : slot_bytes_(slot_bytes) {
    sycl::context& ctx = c10::xpu::get_device_context();
    if (c10::xpu::device_count() < 2) throw std::runtime_error("xpu exchange1p: needs two XPU devices in this process");
    if (ctx.get_devices().size() < 2)
      throw std::runtime_error("xpu exchange1p: torch's SYCL context does not hold both cards");
    for (int c = 0; c < 2; ++c) {
      sycl::device& dev = c10::xpu::get_raw_device(c);
      sycl::queue& q = c10::xpu::getCurrentXPUStream(c).queue();
      for (int b = 0; b < 2; ++b) {
        inbox_[c][b] = static_cast<uint64_t*>(sycl::malloc_host(2 * SMALL, ctx));
        big_[c][b] = static_cast<char*>(sycl::malloc_host(slot_bytes, ctx));
        theirs_dev_[c][b] = static_cast<char*>(sycl::malloc_device(slot_bytes, dev, ctx));
        if (!inbox_[c][b] || !big_[c][b] || !theirs_dev_[c][b]) throw std::runtime_error("xpu exchange1p: USM allocation failed");
        std::memset(inbox_[c][b], 0, 2 * SMALL);
      }
      dseq_[c] = static_cast<int64_t*>(sycl::malloc_device(16 * sizeof(int64_t), dev, ctx));   // [0] seq, [8] done count
      marker_[c] = static_cast<int64_t*>(sycl::malloc_host(16 * sizeof(int64_t), ctx));        // [b * 8], b = 0, 1
      status_[c] = static_cast<int64_t*>(sycl::malloc_host(8 * sizeof(int64_t), ctx));
      if (!dseq_[c] || !marker_[c] || !status_[c]) throw std::runtime_error("xpu exchange1p: USM allocation failed");
      std::memset(marker_[c], 0, 16 * sizeof(int64_t));
      std::memset(status_[c], 0, 8 * sizeof(int64_t));
      q.memset(dseq_[c], 0, 16 * sizeof(int64_t)).wait();
    }
  }
  ~Pair() { close(); }

  at::Tensor all_reduce(const at::Tensor& t) { return run(t, ADD, FUSED, std::nullopt); }
  // flattened gather in card order: the first numel elements are card 0's, the next numel card 1's
  at::Tensor all_gather(const at::Tensor& t) { return run(t, GATHER, FUSED, std::nullopt); }
  // the split form: start() queues my half and returns the (not yet filled) output; finish() queues the wait and combine
  at::Tensor start(const at::Tensor& t, int mode) { return run(t, mode, EXPORT, std::nullopt); }
  at::Tensor finish(const at::Tensor& t, const at::Tensor& out, int mode) { return run(t, mode, FINISH, out); }

  // (small calls, small calls recorded into a graph, big calls) for one card
  std::tuple<int64_t, int64_t, int64_t> counts(int c) const { return {n_small_[c], n_recorded_[c], n_big_[c]}; }
  // (card 0 timeout seq, card 1 timeout seq): zeros mean healthy; negative = big path
  std::tuple<int64_t, int64_t> status() const {
    return {reinterpret_cast<volatile int64_t*>(status_[0])[0], reinterpret_cast<volatile int64_t*>(status_[1])[0]};
  }

  void close() {
    if (closed_) return;
    closed_ = true;
    sycl::context& ctx = c10::xpu::get_device_context();
    for (int c = 0; c < 2; ++c) {
      for (int b = 0; b < 2; ++b) {
        if (inbox_[c][b]) sycl::free(inbox_[c][b], ctx);
        if (big_[c][b]) sycl::free(big_[c][b], ctx);
        if (theirs_dev_[c][b]) sycl::free(theirs_dev_[c][b], ctx);
      }
      if (dseq_[c]) sycl::free(dseq_[c], ctx);
      if (marker_[c]) sycl::free(marker_[c], ctx);
      if (status_[c]) sycl::free(status_[c], ctx);
    }
  }

 private:
  void check_status(int c) const {
    volatile int64_t* s = reinterpret_cast<volatile int64_t*>(status_[c]);
    if (s[0] != 0)
      throw std::runtime_error("xpu exchange1p: card " + std::to_string(c) + " waited too long for the other card's half at "
                               "call #" + std::to_string(s[0]) + " (negative = big path); results after that point are invalid");
  }

  at::Tensor run(const at::Tensor& t, int mode, int phase, const std::optional<at::Tensor>& given) {
    TORCH_CHECK(t.is_xpu(), "xpu exchange1p: tensor must live on an XPU");
    TORCH_CHECK(t.is_contiguous(), "xpu exchange1p: tensor must be contiguous");
    const int c = t.device().index();
    TORCH_CHECK(c == 0 || c == 1, "xpu exchange1p: device index must be 0 or 1");
    check_status(c);
    const size_t bytes = (size_t)t.numel() * t.element_size();
    TORCH_CHECK(bytes <= slot_bytes_, "xpu exchange1p: tensor of ", bytes, " B is larger than a slot (", slot_bytes_, " B)");
    int dt = RAW;
    if (mode == ADD) {
      switch (t.scalar_type()) {
        case at::kFloat: dt = F32; break;
        case at::kBFloat16: dt = BF16; break;
        case at::kHalf: dt = F16; break;
        default: TORCH_CHECK(false, "xpu exchange1p: unsupported dtype for all_reduce ", t.scalar_type());
      }
      TORCH_CHECK(bytes % 4 == 0, "xpu exchange1p: all_reduce byte count must be a multiple of 4");
    }
    at::Tensor out = given.has_value() ? *given : (mode == ADD ? at::empty_like(t) : at::empty({2 * t.numel()}, t.options()));
    if (bytes == 0) return out;
    sycl::queue& q = c10::xpu::getCurrentXPUStream(c).queue();
    const bool recording = q.ext_oneapi_get_state() == oex::queue_state::recording;
    if (bytes <= SMALL && bytes % 4 == 0) {
      if (phase != FINISH) { ++n_small_[c]; if (recording) ++n_recorded_[c]; }
      small(q, c, t, out, bytes / 4, mode, dt, phase);
      return out;
    }
    TORCH_CHECK(!recording, "xpu exchange1p: a ", bytes, " B exchange (over 64 KB) inside an XPU graph capture; the large "
                "path is host-driven. Cap the cudagraph capture sizes so only decode-size batches are captured.");
    if (phase != FINISH) ++n_big_[c];
    big(q, c, t, out, bytes, mode, dt, phase);
    return out;
  }

  void small(sycl::queue& q, int c, const at::Tensor& t, at::Tensor& out, size_t nwords, int mode, int dt, int phase) {
    const int peer = 1 - c;
    size_t g = nwords / WORDS_PER_GROUP;
    const int groups = (int)(g < 1 ? 1 : (g > MAX_GROUPS ? MAX_GROUPS : g));
    const size_t per = (nwords + groups - 1) / groups;
    const uint32_t* x = static_cast<const uint32_t*>(t.data_ptr());
    uint32_t* y = static_cast<uint32_t*>(out.data_ptr());
    uint64_t* o0 = inbox_[peer][0];
    uint64_t* o1 = inbox_[peer][1];
    const uint64_t* i0 = inbox_[c][0];
    const uint64_t* i1 = inbox_[c][1];
    int64_t* dseq = dseq_[c];
    int64_t* status = status_[c];
    // gather: my words at [c * nwords], the peer's at [peer * nwords]
    const size_t mine_off = (size_t)c * nwords, theirs_off = (size_t)peer * nwords;
    auto body = [=](sycl::nd_item<1> it, auto combine) {
      const int64_t s = dseq[0] + 1;
      const uint32_t tag = (uint32_t)s;
      uncached_wptr<uint64_t> w{(s & 1) ? o1 : o0};
      const uint64_t* in = (s & 1) ? i1 : i0;
      const size_t grp = it.get_group(0), lid = it.get_local_id(0);
      const size_t a = grp * per, z = a + per < nwords ? a + per : nwords;
      if (phase != FINISH)
        for (size_t i = a + lid; i < z; i += WG) w[i] = tagged(x[i], tag);
      if (phase == EXPORT) return;   // the sequence advances in FINISH
      for (size_t i = a + lid; i < z; i += WG) combine(i, data_of(wait_tagged(in + i, tag, s, status)));
      sycl::group_barrier(it.get_group());
      if (lid == 0) {
        sycl::atomic_ref<int64_t, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                         sycl::access::address_space::global_space> done(dseq[8]);
        if (done.fetch_add(1) == groups - 1) {
          done.store(0);
          sycl::atomic_ref<int64_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                           sycl::access::address_space::global_space> sq(dseq[0]);
          sq.store(s);
        }
      }
    };
    const sycl::nd_range<1> range(WG * groups, WG);
    if (mode == GATHER) {
      q.parallel_for(range, [=](sycl::nd_item<1> it) {
        body(it, [&](size_t i, uint32_t v) { y[mine_off + i] = x[i]; y[theirs_off + i] = v; });
      });
    } else if (dt == F32) {
      q.parallel_for(range, [=](sycl::nd_item<1> it) { body(it, [&](size_t i, uint32_t v) { y[i] = add_words<F32>(x[i], v); }); });
    } else if (dt == BF16) {
      q.parallel_for(range, [=](sycl::nd_item<1> it) { body(it, [&](size_t i, uint32_t v) { y[i] = add_words<BF16>(x[i], v); }); });
    } else {
      q.parallel_for(range, [=](sycl::nd_item<1> it) { body(it, [&](size_t i, uint32_t v) { y[i] = add_words<F16>(x[i], v); }); });
    }
  }

  void big(sycl::queue& q, int c, const at::Tensor& t, at::Tensor& out, size_t bytes, int mode, int dt, int phase) {
    const int peer = 1 - c;
    // EXPORT / FUSED start a new sequence; FINISH completes the one its EXPORT started (calls alternate strictly per card)
    const int64_t seq = phase == FINISH ? bseq_[c] : ++bseq_[c];
    const int b = (int)(seq & 1);
    const char* x = static_cast<const char*>(t.data_ptr());
    if (phase != FINISH) {
      // my bytes into my pinned slot, then the marker (in order: after the copy)
      q.memcpy(big_[c][b], x, bytes);
      int64_t* mine = marker_[c] + b * 8;
      q.single_task([=]() { uncached_wptr<int64_t> m{mine}; m[0] = seq; });
      if (phase == EXPORT) return;
    }
    // wait for the peer's marker; slot reuse is safe: my copy into slot b at seq+2 runs after my poll at seq+1, which
    // needs the peer's marker seq+1, stored only after the peer's queue has read my slot b at seq
    const int64_t* theirs = marker_[peer] + b * 8;
    int64_t* status = status_[c];
    q.single_task([=]() {
      uncached_rptr<const int64_t> r{theirs};
      uint32_t polls = 0;
      for (;;) {
        int64_t v = r[0];
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
        if (v >= seq) break;
        if (++polls > SPIN_LIMIT) { uncached_wptr<int64_t> s{status}; if (s[0] == 0) s[0] = -seq; break; }
      }
    });
    char* y = static_cast<char*>(out.data_ptr());
    if (mode == GATHER) {
      q.memcpy(y + (size_t)c * bytes, x, bytes);
      q.memcpy(y + (size_t)peer * bytes, big_[peer][b], bytes);
      return;
    }
    q.memcpy(theirs_dev_[c][b], big_[peer][b], bytes);
    const uint32_t* xw = reinterpret_cast<const uint32_t*>(x);
    const uint32_t* tw = reinterpret_cast<const uint32_t*>(theirs_dev_[c][b]);
    uint32_t* yw = reinterpret_cast<uint32_t*>(y);
    const size_t n = bytes / 4;
    switch (dt) {
      case F32: q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { yw[i] = add_words<F32>(xw[i], tw[i]); }); break;
      case BF16: q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { yw[i] = add_words<BF16>(xw[i], tw[i]); }); break;
      default: q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { yw[i] = add_words<F16>(xw[i], tw[i]); }); break;
    }
  }

  size_t slot_bytes_;
  uint64_t* inbox_[2][2] = {};     // pinned host, tagged words: inbox_[c][b] is what card c READS (the other card writes it)
  char* big_[2][2] = {};           // pinned host, plain bytes: big_[c][b] is what card c WRITES (the other card copies it in)
  char* theirs_dev_[2][2] = {};    // card c's memory: the other card's half, copied in (big path, all_reduce)
  int64_t* dseq_[2] = {};          // card c's memory: [0] last small sequence completed, [8] groups done in the current call
  int64_t* marker_[2] = {};        // pinned host: card c's big-path marker per slot parity, at [b * 8]
  int64_t* status_[2] = {};        // pinned host: [0] the call card c gave up waiting at
  int64_t bseq_[2] = {0, 0};
  int64_t n_small_[2] = {0, 0}, n_big_[2] = {0, 0}, n_recorded_[2] = {0, 0};
  bool closed_ = false;
};

// The baton's hand-over word (xpu_oneproc.Baton). A Python condition variable took ~85 us per hand-over on Windows
// (probe/oneproc-exchange-torch, 2026-10-13: 56 per eager decode step = 4.8 ms); here the waiting thread spins without
// the GIL, so the running thread never contends for it and the hand-over costs about one cache-line transfer. Waits
// longer than 2 s (a model load on the other rank) back off to Sleep(1) so an idle rank does not burn a core forever.
class Turn {
 public:
  int get() const { return holder_.load(std::memory_order_acquire); }
  void set(int h) { holder_.store(h, std::memory_order_release); }
  // set the holder to `other`, then wait as `me`
  void pass_and_wait(int me, int other) { holder_.store(other, std::memory_order_release); wait_for(me); }
  // return once this thread holds the turn: the holder is `me`, or nobody (-1) and this thread claims it
  void wait_for(int me) {
    auto t0 = std::chrono::steady_clock::now();
    uint64_t n = 0;
    for (;;) {
      int h = holder_.load(std::memory_order_acquire);
      if (h == me) return;
      if (h == -1 && holder_.compare_exchange_weak(h, me, std::memory_order_acq_rel)) return;
      ++n;
      if (n < 20000) { _mm_pause(); continue; }
      if ((n & 1023) == 0 && std::chrono::steady_clock::now() - t0 > std::chrono::seconds(2)) Sleep(1);
      else SwitchToThread();
    }
  }

 private:
  std::atomic<int> holder_{-1};
};

}  // namespace

PYBIND11_MODULE(_xpu_exchange1p, m) {
  pybind11::class_<Turn>(m, "Turn")
      .def(pybind11::init<>())
      .def("get", &Turn::get)
      .def("set", &Turn::set, pybind11::arg("holder"))
      .def("pass_and_wait", &Turn::pass_and_wait, pybind11::arg("me"), pybind11::arg("other"),
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("wait_for", &Turn::wait_for, pybind11::arg("me"), pybind11::call_guard<pybind11::gil_scoped_release>());
  m.doc() = "two-card all_reduce / all_gather for a one-process two-card vLLM on Windows XPU: tagged words through pinned "
            "host memory, read and written by both cards' kernels directly";
  pybind11::class_<Pair>(m, "Pair")
      .def(pybind11::init<size_t>(), pybind11::arg("slot_bytes"))
      .def("all_reduce", &Pair::all_reduce, pybind11::arg("tensor"),
           "sum with the other card's tensor of the same call, on this tensor's card's current stream; returns at once")
      .def("all_gather", &Pair::all_gather, pybind11::arg("tensor"),
           "flattened [card 0's elements, card 1's elements] on this tensor's card; returns at once")
      .def("start", &Pair::start, pybind11::arg("tensor"), pybind11::arg("mode"),
           "split form, first half: queue my half (mode 0 all_reduce, 1 all_gather); returns the output, filled by finish")
      .def("finish", &Pair::finish, pybind11::arg("tensor"), pybind11::arg("out"), pybind11::arg("mode"),
           "split form, second half: queue the wait for the other card's half and the combine into `out`")
      .def("counts", &Pair::counts, pybind11::arg("card"), "(small calls, small calls recorded into a graph, big calls)")
      .def("status", &Pair::status, "(card 0 timeout call, card 1 timeout call); zeros mean healthy")
      .def("close", &Pair::close);
}
