#pragma once

namespace evk::metal {
inline constexpr char BLIT_SHADER[] = R"msl(
#include <metal_stdlib>
using namespace metal;

struct BlitParams {
    float4 origin;
    float4 scale;
    uint mip;
    uint layer;
};
struct BlitVertex { float4 position [[position]]; };
struct BlitDepth { float value [[depth(any)]]; };

vertex BlitVertex blit_vertex(uint id [[vertex_id]]) {
    const float2 positions[] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return {float4(positions[id], 0, 1)};
}

fragment float4 blit_float_2d(BlitVertex in [[stage_in]], constant BlitParams& p [[buffer(0)]],
        texture2d<float> image [[texture(0)]], sampler filter [[sampler(0)]]) {
    float2 uv = p.origin.xy + in.position.xy * p.scale.xy;
    return image.sample(filter, uv, level(p.mip));
}
fragment float4 blit_float_array(BlitVertex in [[stage_in]], constant BlitParams& p [[buffer(0)]],
        texture2d_array<float> image [[texture(0)]], sampler filter [[sampler(0)]]) {
    float2 uv = p.origin.xy + in.position.xy * p.scale.xy;
    return image.sample(filter, uv, p.layer, level(p.mip));
}
fragment float4 blit_float_3d(BlitVertex in [[stage_in]], constant BlitParams& p [[buffer(0)]],
        texture3d<float> image [[texture(0)]], sampler filter [[sampler(0)]]) {
    float3 uv = p.origin.xyz + float3(in.position.xy, 0) * p.scale.xyz;
    return image.sample(filter, uv, level(p.mip));
}

template<typename T> uint2 blit_pixel(float2 uv, T image, uint mip) {
    uint2 size = uint2(image.get_width(mip), image.get_height(mip));
    return uint2(clamp(floor(uv * float2(size)), float2(0), float2(size - 1)));
}
template<typename T> uint3 blit_voxel(float3 uv, T image, uint mip) {
    uint3 size = uint3(image.get_width(mip), image.get_height(mip), image.get_depth(mip));
    return uint3(clamp(floor(uv * float3(size)), float3(0), float3(size - 1)));
}

fragment uint4 blit_uint_2d(BlitVertex in [[stage_in]], constant BlitParams& p [[buffer(0)]],
        texture2d<uint> image [[texture(0)]]) {
    return image.read(blit_pixel(p.origin.xy + in.position.xy * p.scale.xy, image, p.mip), p.mip);
}
fragment uint4 blit_uint_array(BlitVertex in [[stage_in]], constant BlitParams& p [[buffer(0)]],
        texture2d_array<uint> image [[texture(0)]]) {
    return image.read(blit_pixel(p.origin.xy + in.position.xy * p.scale.xy, image, p.mip), p.layer, p.mip);
}
fragment uint4 blit_uint_3d(BlitVertex in [[stage_in]], constant BlitParams& p [[buffer(0)]],
        texture3d<uint> image [[texture(0)]]) {
    return image.read(blit_voxel(p.origin.xyz + float3(in.position.xy, 0) * p.scale.xyz, image, p.mip), p.mip);
}
fragment int4 blit_int_2d(BlitVertex in [[stage_in]], constant BlitParams& p [[buffer(0)]],
        texture2d<int> image [[texture(0)]]) {
    return image.read(blit_pixel(p.origin.xy + in.position.xy * p.scale.xy, image, p.mip), p.mip);
}
fragment int4 blit_int_array(BlitVertex in [[stage_in]], constant BlitParams& p [[buffer(0)]],
        texture2d_array<int> image [[texture(0)]]) {
    return image.read(blit_pixel(p.origin.xy + in.position.xy * p.scale.xy, image, p.mip), p.layer, p.mip);
}
fragment int4 blit_int_3d(BlitVertex in [[stage_in]], constant BlitParams& p [[buffer(0)]],
        texture3d<int> image [[texture(0)]]) {
    return image.read(blit_voxel(p.origin.xyz + float3(in.position.xy, 0) * p.scale.xyz, image, p.mip), p.mip);
}

fragment BlitDepth blit_depth_2d(BlitVertex in [[stage_in]], constant BlitParams& p [[buffer(0)]],
        depth2d<float> image [[texture(0)]]) {
    return {image.read(blit_pixel(p.origin.xy + in.position.xy * p.scale.xy, image, p.mip), p.mip)};
}
fragment BlitDepth blit_depth_array(BlitVertex in [[stage_in]], constant BlitParams& p [[buffer(0)]],
        depth2d_array<float> image [[texture(0)]]) {
    return {image.read(blit_pixel(p.origin.xy + in.position.xy * p.scale.xy, image, p.mip), p.layer, p.mip)};
}
)msl";
}
