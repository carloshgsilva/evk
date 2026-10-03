#pragma once

namespace evk::metal {
inline constexpr const char* INDIRECT_COUNT_SHADER = R"msl(
#include <metal_stdlib>
using namespace metal;

struct DrawFilter { uint stride; uint words; uint maximum; };
kernel void filter_draw_count(device const uint* source [[buffer(0)]],
        device const uint& count [[buffer(1)]], device uint* target [[buffer(2)]],
        constant DrawFilter& params [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    if (index >= params.maximum) return;
    bool active = index < min(count, params.maximum);
    for (uint word = 0; word < params.words; ++word) {
        target[index * params.words + word] = active ? source[index * params.stride + word] : 0u;
    }
}
)msl";
}
