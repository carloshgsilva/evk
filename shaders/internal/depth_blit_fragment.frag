#version 460

layout(binding = 0) uniform sampler2D sourceDepth;
layout(push_constant) uniform Region {
    vec4 source;
    vec4 destination;
} region;

void main() {
    vec2 position = region.source.xy
        + (gl_FragCoord.xy - region.destination.xy) * region.source.zw / region.destination.zw;
    gl_FragDepth = texelFetch(sourceDepth, ivec2(floor(position)), 0).r;
}
