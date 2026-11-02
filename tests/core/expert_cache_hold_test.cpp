// STRATA_BATCH_LAZY: holding the expert cache's tail changes which slots are usable, and nothing else: the slots keep
// their addresses and bytes, the VRAM stays mapped, and releasing the tail makes every slot usable again.
// Needs a CUDA device (exits 77 without one).
#include "strata/core/expert_cache.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace {
int fail(const char* what, const std::string& err = "") {
    std::fprintf(stderr, "FAIL %s %s\n", what, err.c_str());
    return 1;
}
}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::puts("no CUDA device: skipped");
        return 77;
    }
    constexpr int64_t kBlob = 1 << 20, kSlots = 16, kKeep = 6;
    strata::core::ExpertCache c;
    std::string err;
    if (!c.open(kSlots, 2, 8, kBlob, err)) return fail("open", err);
    uint8_t* const first = c.device_slot(0);
    std::vector<uint8_t> b((size_t) kBlob);
    for (int64_t i = 0; i < kSlots; ++i) {
        std::fill(b.begin(), b.end(), (uint8_t) (i * 7 + 1));
        if (cudaMemcpy(c.device_slot((int32_t) i), b.data(), (size_t) kBlob, cudaMemcpyHostToDevice) != cudaSuccess)
            return fail("fill");
    }
    c.hold_live_slots(kKeep);
    if (c.slots() != kKeep || c.bytes() != kKeep * kBlob) return fail("held sizes");
    if (c.full_slots() != kSlots || c.full_bytes() != kSlots * kBlob) return fail("full sizes while held");
    if (c.device_slot(0) != first) return fail("the arena moved");
    // the tail stays mapped: what the slots' sessions write there is really there
    uint8_t* const tail = first + kKeep * kBlob;
    std::fill(b.begin(), b.end(), (uint8_t) 0xAB);
    if (cudaMemcpy(tail, b.data(), (size_t) kBlob, cudaMemcpyHostToDevice) != cudaSuccess) return fail("write the tail");
    std::fill(b.begin(), b.end(), (uint8_t) 0);
    if (cudaMemcpy(b.data(), tail, (size_t) kBlob, cudaMemcpyDeviceToHost) != cudaSuccess) return fail("read the tail");
    for (const uint8_t x : b)
        if (x != 0xAB) return fail("the tail's bytes");
    // the slots kept are untouched
    for (int64_t i = 0; i < kKeep; ++i) {
        if (cudaMemcpy(b.data(), c.device_slot((int32_t) i), (size_t) kBlob, cudaMemcpyDeviceToHost) != cudaSuccess)
            return fail("read a slot");
        for (const uint8_t x : b)
            if (x != (uint8_t) (i * 7 + 1)) return fail("a slot kept changed");
    }
    c.hold_live_slots(kSlots + 5);
    if (c.slots() != kSlots) return fail("hold past the end");
    c.hold_live_slots(kKeep);
    c.release_live_slots();
    if (c.slots() != kSlots || c.bytes() != kSlots * kBlob || c.device_slot(0) != first) return fail("release");
    c.close();
    std::puts("expert cache hold: the tail is usable while held, the slots keep their place, release gives all back");
    return 0;
}
