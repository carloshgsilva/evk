#pragma once

namespace evk::metal {
inline constexpr char TRIANGLE_DATA_SHADER[] = R"msl(
#include <metal_stdlib>
using namespace metal;

struct TriangleDataParams { uint stride; uint count; };
kernel void triangle_data(const device uchar* vertices [[buffer(0)]],
        const device uint* indices [[buffer(1)]], device packed_float3* positions [[buffer(2)]],
        constant TriangleDataParams& params [[buffer(3)]], uint triangle [[thread_position_in_grid]]) {
    if (triangle >= params.count) return;
    for (uint corner = 0; corner < 3; ++corner) {
        uint index = indices[triangle * 3 + corner];
        positions[triangle * 3 + corner] = *reinterpret_cast<const device packed_float3*>(vertices + index * params.stride);
    }
}
)msl";
}
