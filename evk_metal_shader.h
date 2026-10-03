#pragma once

#include <cstdint>
#include <span>

namespace evk::metal {
constexpr uint32_t BUFFER_COUNT = 16384;
constexpr uint32_t IMAGE_COUNT = 16384;
constexpr uint32_t TLAS_COUNT = 16384;
constexpr uint32_t STORAGE_ID = 0;
constexpr uint32_t TEXTURE_ID = BUFFER_COUNT;
constexpr uint32_t SAMPLER_ID = TEXTURE_ID + IMAGE_COUNT;
constexpr uint32_t IMAGE_ID = SAMPLER_ID + IMAGE_COUNT;
constexpr uint32_t TLAS_ID = IMAGE_ID + IMAGE_COUNT;
constexpr uint32_t ARGUMENT_BUFFER = 0;
constexpr uint32_t PUSH_BUFFER = 1;
constexpr uint32_t VERTEX_BUFFER = 2;
enum class ShaderStage : uint32_t { Vertex, Fragment, Compute };
enum class ConstantType : uint32_t { Unused, Bool, Int, UInt, Float };

inline uint64_t ShaderHash(std::span<const uint8_t> bytes) {
    uint64_t hash = 14695981039346656037ull;
    for (uint8_t byte : bytes) {
        hash = (hash ^ byte) * 1099511628211ull;
    }
    return hash;
}

struct ShaderInfo {
    uint32_t magic = 0x4D534C31;
    uint32_t version = 5;
    ShaderStage stage = ShaderStage::Vertex;
    uint32_t groupX = 1;
    uint32_t groupY = 1;
    uint32_t groupZ = 1;
    uint32_t groupConstantX = UINT32_MAX;
    uint32_t groupConstantY = UINT32_MAX;
    uint32_t groupConstantZ = UINT32_MAX;
    uint64_t hash = 0;
    ConstantType constantTypes[32] = {};
    uint32_t constantDefaults[32] = {};
    uint32_t staticConstants = 0;
};
}
