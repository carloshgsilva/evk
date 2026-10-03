#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <algorithm>
#include <array>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <type_traits>
#include <vector>

#include "evk.h"
#include "evk_metal_blit.h"
#include "evk_metal_indirect.h"
#include "evk_metal_ray.h"
#include "evk_metal_shader.h"

namespace evk {
namespace {
constexpr uint32_t COMMAND_COUNT = 4;
constexpr uint32_t SAMPLE_COUNT = 1024;
constexpr uint32_t TIMESTAMP_COUNT = 128;
constexpr uint64_t STAGING_BYTES = 64'000'000;

void Require(bool condition, const char* format, ...) {
    if (condition) return;
    va_list args;
    va_start(args, format);
    std::fprintf(stderr, "[evk] Metal: ");
    std::vfprintf(stderr, format, args);
    std::fprintf(stderr, "\n");
    va_end(args);
    std::abort();
}

template<class T> bool Has(T flags, T flag) { return (uint32_t(flags) & uint32_t(flag)) != 0; }
bool Depth(Format format) { return format == Format::D32Sfloat || format == Format::D24UnormS8Uint; }
bool UInt(Format format) { return format == Format::R8Uint || format == Format::R16Uint || format == Format::R32Uint || format == Format::RGBA32Uint; }

MTLPixelFormat PixelFormat(Format format) {
    switch (format) {
        case Format::R8Uint: return MTLPixelFormatR8Uint;
        case Format::R16Uint: return MTLPixelFormatR16Uint;
        case Format::R32Uint: return MTLPixelFormatR32Uint;
        case Format::BGRA8Unorm: return MTLPixelFormatBGRA8Unorm;
        case Format::RGBA8Unorm: return MTLPixelFormatRGBA8Unorm;
        case Format::RGBA8Snorm: return MTLPixelFormatRGBA8Snorm;
        case Format::RG16Sfloat: return MTLPixelFormatRG16Float;
        case Format::RGBA16Sfloat: return MTLPixelFormatRGBA16Float;
        case Format::RGBA16Unorm: return MTLPixelFormatRGBA16Unorm;
        case Format::RGBA16Snorm: return MTLPixelFormatRGBA16Snorm;
        case Format::R32Sfloat: return MTLPixelFormatR32Float;
        case Format::RG32Sfloat: return MTLPixelFormatRG32Float;
        case Format::RGBA32Sfloat: return MTLPixelFormatRGBA32Float;
        case Format::RGBA32Sint: return MTLPixelFormatRGBA32Sint;
        case Format::RGBA32Uint: return MTLPixelFormatRGBA32Uint;
        case Format::D32Sfloat: return MTLPixelFormatDepth32Float;
        case Format::D24UnormS8Uint: return MTLPixelFormatDepth24Unorm_Stencil8;
        default: Require(false, "Unsupported texture format %u", uint32_t(format)); return MTLPixelFormatInvalid;
    }
}

uint32_t PixelBytes(Format format) {
    switch (format) {
        case Format::R8Uint: return 1;
        case Format::R16Uint: return 2;
        case Format::RGBA16Sfloat: case Format::RGBA16Unorm: case Format::RGBA16Snorm: case Format::RG32Sfloat: return 8;
        case Format::RGB32Sfloat: return 12;
        case Format::RGBA32Sfloat: case Format::RGBA32Sint: case Format::RGBA32Uint: return 16;
        default: return 4;
    }
}

MTLVertexFormat VertexFormat(Format format) {
    switch (format) {
        case Format::R32Sfloat: return MTLVertexFormatFloat;
        case Format::RG32Sfloat: return MTLVertexFormatFloat2;
        case Format::RGB32Sfloat: return MTLVertexFormatFloat3;
        case Format::RGBA32Sfloat: return MTLVertexFormatFloat4;
        case Format::RGBA8Unorm: return MTLVertexFormatUChar4Normalized;
        case Format::RGBA8Snorm: return MTLVertexFormatChar4Normalized;
        case Format::RGBA32Uint: return MTLVertexFormatUInt4;
        case Format::RGBA32Sint: return MTLVertexFormatInt4;
        default: Require(false, "Unsupported vertex format %u", uint32_t(format)); return MTLVertexFormatInvalid;
    }
}

struct Slots {
    uint32_t limit = 0;
    uint32_t next = 0;
    std::vector<int> free;
    void Initialize(uint32_t count) { limit = count; free.reserve(count); }
    int Allocate() {
        if (!free.empty()) { int value = free.back(); free.pop_back(); return value; }
        Require(next < limit, "Bindless resource limit exceeded (%u)", limit);
        return int(next++);
    }
    void Release(int value) { free.push_back(value); }
};

struct MetalBuffer : Resource {
    BufferDesc desc;
    id<MTLBuffer> buffer = nil;
    ~MetalBuffer();
};
struct MetalImage : Resource {
    ImageDesc desc;
    id<MTLTexture> texture = nil;
    id<MTLTexture> blitTexture = nil;
    ~MetalImage();
};
struct MetalPipeline : Resource {
    PipelineDesc desc;
    id<MTLRenderPipelineState> render = nil;
    id<MTLComputePipelineState> compute = nil;
    id<MTLDepthStencilState> depth = nil;
    MTLSize group = {1, 1, 1};
    bool usesDrawID = false;
};
struct MetalBLAS : Resource {
    BLASDesc geometry;
    MTLPrimitiveAccelerationStructureDescriptor* desc = nil;
    id<MTLAccelerationStructure> acceleration = nil;
    Buffer triangleData;
    uint64_t scratchBytes = 0;
    bool built = false;
    ~MetalBLAS();
};
struct MetalTLAS : Resource {
    MTLIndirectInstanceAccelerationStructureDescriptor* desc = nil;
    id<MTLAccelerationStructure> acceleration = nil;
    Buffer instances;
    Buffer count;
    std::vector<BLAS> references;
    uint64_t scratchBytes = 0;
    uint32_t instanceCount = 0;
    bool allowUpdate = false;
    bool built = false;
    ~MetalTLAS();
};

struct Marker { const char* name = nullptr; uint32_t first = 0; uint32_t last = 0; };
struct Command {
    Cmd api;
    id<MTLCommandBuffer> buffer = nil;
    id<MTLRenderCommandEncoder> render = nil;
    id<MTLComputeCommandEncoder> compute = nil;
    id<MTLBlitCommandEncoder> blit = nil;
    id<MTLAccelerationStructureCommandEncoder> acceleration = nil;
    MTLRenderPassDescriptor* renderPass = nil;
    std::array<MTLStoreAction, MAX_ATTACHMENTS_COUNT> colorStores = {};
    MTLStoreAction depthStore = MTLStoreActionDontCare;
    MTLStoreAction stencilStore = MTLStoreActionDontCare;
    MTLViewport viewport = {};
    MTLScissorRect scissor = {};
    MTLComputePassDescriptor* computePass = nil;
    MTLBlitPassDescriptor* blitPass = nil;
    MTLAccelerationStructurePassDescriptor* accelerationPass = nil;
    id<MTLCounterSampleBuffer> counters = nil;
    Buffer staging;
    uint64_t stagingOffset = 0;
    uint64_t scratchOffset = 0;
    uint64_t scratchBytes = 0;
    uint64_t submission = 0;
    bool recording = false;
    bool pending = false;
    Pipeline pipeline;
    Buffer vertices;
    Buffer indices;
    uint64_t vertexOffset = 0;
    uint64_t indexOffset = 0;
    bool halfIndices = false;
    std::array<uint8_t, 256> push = {};
    std::array<Marker, TIMESTAMP_COUNT> markers;
    uint32_t markerCount = 0;
    uint32_t samples = 0;
    id<CAMetalDrawable> drawable = nil;
    Image drawableImage;
};

struct State {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLArgumentEncoder> arguments = nil;
    id<MTLBuffer> argumentBuffer = nil;
    id<MTLSamplerState> nearest = nil;
    id<MTLSamplerState> linear = nil;
    id<MTLLibrary> blitLibrary = nil;
    id<MTLDepthStencilState> blitDepth = nil;
    id<MTLDepthStencilState> blitColorDepth = nil;
    id<MTLComputePipelineState> triangleData = nil;
    id<MTLComputePipelineState> indirectCount = nil;
    std::array<std::array<id<MTLRenderPipelineState>, 3>, uint32_t(Format::D32Sfloat) + 1> blitPipelines;
    CAMetalLayer* layer = nil;
    NSView* view = nil;
    Features features;
    Slots buffers;
    Slots images;
    Slots tlas;
    std::array<Command, COMMAND_COUNT> commands;
    std::vector<id<MTLResource>> reads;
    std::vector<id<MTLResource>> writes;
    std::vector<std::pair<uint64_t, Resource*>> deletions;
    std::vector<TimestampEntry> timestamps;
    uint64_t nextSubmission = 1;
    uint64_t completed = 0;
    uint64_t liveBytes = 0;
    uint32_t allocations = 0;
    bool shuttingDown = false;
};
State* G = nullptr;
State& S() { Require(G != nullptr, "Backend is not initialized"); return *G; }
MetalBuffer& B(const Buffer& buffer) { Require(bool(buffer), "Invalid buffer"); return *static_cast<MetalBuffer*>(buffer.res); }
MetalImage& I(const Image& image) { Require(bool(image), "Invalid image"); return *static_cast<MetalImage*>(image.res); }
MetalPipeline& P(const Pipeline& pipeline) { Require(bool(pipeline), "Invalid pipeline"); return *static_cast<MetalPipeline*>(pipeline.res); }
MetalBLAS& A(const BLAS& blas) { Require(bool(blas), "Invalid BLAS"); return *static_cast<MetalBLAS*>(blas.res); }
MetalTLAS& T(const TLAS& tlas) { Require(bool(tlas), "Invalid TLAS"); return *static_cast<MetalTLAS*>(tlas.res); }
Command& C(const Cmd& cmd) { Require(cmd._internal != nullptr, "Invalid command buffer"); return *static_cast<Command*>(cmd._internal); }

void AddResource(id<MTLResource> resource, bool writable) {
    auto& list = writable ? S().writes : S().reads;
    Require(list.size() < list.capacity(), "Resource residency capacity exceeded");
    list.push_back(resource);
    S().liveBytes += resource.allocatedSize;
    ++S().allocations;
}
void RemoveResource(id<MTLResource> resource) {
    for (auto* list : {&S().reads, &S().writes}) {
        auto found = std::find(list->begin(), list->end(), resource);
        if (found == list->end()) continue;
        *found = list->back(); list->pop_back();
        S().liveBytes -= resource.allocatedSize;
        --S().allocations;
        return;
    }
}
MetalBuffer::~MetalBuffer() {
    if (resourceid >= 0) {
        [S().arguments setBuffer:nil offset:0 atIndex:metal::STORAGE_ID + resourceid];
        S().buffers.Release(resourceid);
    }
    RemoveResource(buffer);
}
MetalImage::~MetalImage() {
    if (resourceid >= 0) {
        [S().arguments setTexture:nil atIndex:metal::TEXTURE_ID + resourceid];
        [S().arguments setTexture:nil atIndex:metal::IMAGE_ID + resourceid];
        [S().arguments setSamplerState:nil atIndex:metal::SAMPLER_ID + resourceid];
        S().images.Release(resourceid);
    }
    if (texture) RemoveResource(texture);
}
MetalBLAS::~MetalBLAS() { RemoveResource(acceleration); }
MetalTLAS::~MetalTLAS() {
    [S().arguments setAccelerationStructure:nil atIndex:metal::TLAS_ID + resourceid];
    S().tlas.Release(resourceid);
    RemoveResource(acceleration);
}

void Retire() {
    auto& list = S().deletions;
    for (size_t n = 0; n < list.size();) {
        if (!S().shuttingDown && list[n].first > S().completed) { ++n; continue; }
        Resource* resource = list[n].second;
        list[n] = list.back(); list.pop_back();
        delete resource;
    }
}
void EndRenderEncoder(Command& cmd, bool preserve) {
    for (uint32_t index = 0; index < MAX_ATTACHMENTS_COUNT; ++index) {
        auto attachment = cmd.renderPass.colorAttachments[index];
        if (!attachment.texture) continue;
        auto store = attachment.resolveTexture ? MTLStoreActionStoreAndMultisampleResolve : MTLStoreActionStore;
        [cmd.render setColorStoreAction:preserve ? store : cmd.colorStores[index] atIndex:index];
    }
    if (cmd.renderPass.depthAttachment.texture) [cmd.render setDepthStoreAction:preserve ? MTLStoreActionStore : cmd.depthStore];
    if (cmd.renderPass.stencilAttachment.texture) [cmd.render setStencilStoreAction:preserve ? MTLStoreActionStore : cmd.stencilStore];
    [cmd.render endEncoding];
    cmd.render = nil;
}
void EndEncoder(Command& cmd) {
    if (cmd.render) EndRenderEncoder(cmd, false);
    if (cmd.compute) [cmd.compute endEncoding];
    if (cmd.blit) [cmd.blit endEncoding];
    if (cmd.acceleration) [cmd.acceleration endEncoding];
    cmd.render = nil; cmd.compute = nil; cmd.blit = nil;
    cmd.acceleration = nil;
}
uint32_t EncoderSamples(Command& cmd) {
    if (!cmd.counters) return 0;
    Require(cmd.samples + 4 <= SAMPLE_COUNT, "Encoder timestamp capacity exceeded");
    uint32_t first = cmd.samples;
    cmd.samples += 4;
    return first;
}
void Residency(id<MTLComputeCommandEncoder> encoder) {
    auto& state = S();
    if (!state.reads.empty()) [encoder useResources:state.reads.data() count:state.reads.size() usage:MTLResourceUsageRead];
    if (!state.writes.empty()) [encoder useResources:state.writes.data() count:state.writes.size() usage:MTLResourceUsageRead | MTLResourceUsageWrite];
}
void Residency(id<MTLRenderCommandEncoder> encoder) {
    auto& state = S();
    if (!state.reads.empty()) [encoder useResources:state.reads.data() count:state.reads.size() usage:MTLResourceUsageRead stages:MTLRenderStageVertex | MTLRenderStageFragment];
    if (!state.writes.empty()) [encoder useResources:state.writes.data() count:state.writes.size() usage:MTLResourceUsageRead | MTLResourceUsageWrite stages:MTLRenderStageVertex | MTLRenderStageFragment];
}
void Compute(Command& cmd) {
    if (cmd.compute) return;
    Require(!cmd.render, "Compute dispatch inside render pass");
    EndEncoder(cmd);
    @autoreleasepool {
        auto attachment = cmd.computePass.sampleBufferAttachments[0];
        if (cmd.counters) {
            uint32_t first = EncoderSamples(cmd);
            attachment.sampleBuffer = cmd.counters;
            attachment.startOfEncoderSampleIndex = first;
            attachment.endOfEncoderSampleIndex = first + 3;
        }
        cmd.compute = [cmd.buffer computeCommandEncoderWithDescriptor:cmd.computePass];
        [cmd.compute setBuffer:S().argumentBuffer offset:0 atIndex:metal::ARGUMENT_BUFFER];
        Residency(cmd.compute);
    }
}
void Blit(Command& cmd) {
    if (cmd.blit) return;
    Require(!cmd.render, "Transfer inside render pass");
    EndEncoder(cmd);
    @autoreleasepool {
        auto attachment = cmd.blitPass.sampleBufferAttachments[0];
        if (cmd.counters) {
            uint32_t first = EncoderSamples(cmd);
            attachment.sampleBuffer = cmd.counters;
            attachment.startOfEncoderSampleIndex = first;
            attachment.endOfEncoderSampleIndex = first + 3;
        }
        cmd.blit = [cmd.buffer blitCommandEncoderWithDescriptor:cmd.blitPass];
    }
}
void Push(Command& cmd) {
    if (cmd.render) {
        [cmd.render setVertexBytes:cmd.push.data() length:cmd.push.size() atIndex:metal::PUSH_BUFFER];
        [cmd.render setFragmentBytes:cmd.push.data() length:cmd.push.size() atIndex:metal::PUSH_BUFFER];
    }
    if (cmd.compute) [cmd.compute setBytes:cmd.push.data() length:cmd.push.size() atIndex:metal::PUSH_BUFFER];
}
uint64_t StageUpload(Command& cmd, const void* bytes, uint64_t size) {
    uint64_t offset = (cmd.stagingOffset + 255) & ~uint64_t(255);
    Require(offset <= STAGING_BYTES && size <= STAGING_BYTES - offset, "Staging buffer capacity exceeded");
    std::memcpy(static_cast<uint8_t*>(cmd.staging.GetPtr()) + offset, bytes, size);
    cmd.stagingOffset = offset + size;
    return offset;
}

uint64_t Scratch(Command& cmd, uint64_t size) {
    if (size <= cmd.scratchBytes) return cmd.scratchOffset;
    uint64_t offset = (cmd.stagingOffset + 255) & ~uint64_t(255);
    Require(offset <= STAGING_BYTES && size <= STAGING_BYTES - offset, "Acceleration structure scratch exceeds the command staging budget");
    cmd.scratchOffset = offset; cmd.scratchBytes = size;
    cmd.stagingOffset = offset + size;
    return offset;
}

void BuildAcceleration(Command& cmd, id<MTLAccelerationStructure> acceleration, MTLAccelerationStructureDescriptor* desc, uint64_t scratchBytes, bool update) {
    Require(!cmd.render, "Acceleration structure build inside a render pass");
    EndEncoder(cmd);
    uint64_t offset = Scratch(cmd, scratchBytes);
    auto attachment = cmd.accelerationPass.sampleBufferAttachments[0];
    if (cmd.counters) {
        uint32_t first = EncoderSamples(cmd);
        attachment.sampleBuffer = cmd.counters;
        attachment.startOfEncoderSampleIndex = first;
        attachment.endOfEncoderSampleIndex = first + 3;
    }
    cmd.acceleration = [cmd.buffer accelerationStructureCommandEncoderWithDescriptor:cmd.accelerationPass];
    Require(cmd.acceleration != nil, "Cannot create acceleration structure encoder");
    auto& state = S();
    if (!state.reads.empty()) [cmd.acceleration useResources:state.reads.data() count:state.reads.size() usage:MTLResourceUsageRead];
    if (!state.writes.empty()) [cmd.acceleration useResources:state.writes.data() count:state.writes.size() usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    if (update) {
        [cmd.acceleration refitAccelerationStructure:acceleration descriptor:desc destination:acceleration scratchBuffer:B(cmd.staging).buffer scratchBufferOffset:offset];
    } else {
        [cmd.acceleration buildAccelerationStructure:acceleration descriptor:desc scratchBuffer:B(cmd.staging).buffer scratchBufferOffset:offset];
    }
    EndEncoder(cmd);
}

void PrepareTriangleDataPipeline() {
    if (S().triangleData) return;
    NSError* error = nil;
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion3_0;
    auto library = [S().device newLibraryWithSource:[[NSString alloc] initWithUTF8String:metal::TRIANGLE_DATA_SHADER] options:options error:&error];
    Require(library != nil, "Cannot compile triangle-data shader: %s", error.localizedDescription.UTF8String);
    S().triangleData = [S().device newComputePipelineStateWithFunction:[library newFunctionWithName:@"triangle_data"] error:&error];
    Require(S().triangleData != nil, "Cannot create triangle-data pipeline: %s", error.localizedDescription.UTF8String);
}

void ReadTimestamps(Command& cmd) {
    S().timestamps.clear();
    if (!cmd.counters || !cmd.samples || !cmd.markerCount) return;
    @autoreleasepool {
        NSData* data = [cmd.counters resolveCounterRange:NSMakeRange(0, cmd.samples)];
        Require(data.length >= cmd.samples * sizeof(MTLCounterResultTimestamp), "Timestamp resolution failed");
        const auto* samples = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
        uint64_t origin = 0;
        for (uint32_t index = 0; index < cmd.markerCount; ++index) {
            const auto& marker = cmd.markers[index];
            if (marker.last <= marker.first) continue;
            uint64_t start = samples[marker.first].timestamp;
            uint64_t end = samples[marker.last - 1].timestamp;
            if (start == MTLCounterErrorValue || end == MTLCounterErrorValue || end < start) continue;
            if (!origin) origin = start;
            S().timestamps.push_back({double(start - origin) * 1e-6, double(end - origin) * 1e-6, marker.name});
        }
    }
}
void Complete(Command& cmd) {
    Require(cmd.buffer.status != MTLCommandBufferStatusError, "GPU submission failed: %s", cmd.buffer.error.localizedDescription.UTF8String);
    S().completed = std::max(S().completed, cmd.submission);
    ReadTimestamps(cmd);
    cmd.pending = false;
    cmd.pipeline.release(); cmd.vertices.release(); cmd.indices.release();
    if (cmd.drawableImage) I(cmd.drawableImage).texture = nil;
    cmd.drawable = nil;
    Retire();
}
void ResizeLayer() {
    auto& state = S();
    if (!state.layer) return;
    CGFloat scale = state.view.window.backingScaleFactor;
    CGSize size = state.view.bounds.size;
    state.layer.frame = state.view.bounds;
    state.layer.contentsScale = scale;
    state.layer.drawableSize = CGSizeMake(std::max(1.0, size.width * scale), std::max(1.0, size.height * scale));
}

uint32_t BlitType(Format format) {
    if (Depth(format)) return 3;
    if (UInt(format)) return 1;
    if (format == Format::RGBA32Sint) return 2;
    return 0;
}
void PrepareBlitPipelines(Format format) {
    auto& state = S();
    auto& pipelines = state.blitPipelines[uint32_t(format)];
    if (pipelines[0]) return;
    NSError* error = nil;
    if (!state.blitLibrary) {
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_0;
        options.preserveInvariance = YES;
        NSString* source = [[NSString alloc] initWithUTF8String:metal::BLIT_SHADER];
        state.blitLibrary = [state.device newLibraryWithSource:source options:options error:&error];
        Require(state.blitLibrary != nil, "Cannot compile blit shaders: %s", error.localizedDescription.UTF8String);
        MTLDepthStencilDescriptor* depth = [MTLDepthStencilDescriptor new];
        depth.depthCompareFunction = MTLCompareFunctionAlways;
        depth.depthWriteEnabled = YES;
        state.blitDepth = [state.device newDepthStencilStateWithDescriptor:depth];
        depth.depthWriteEnabled = NO;
        state.blitColorDepth = [state.device newDepthStencilStateWithDescriptor:depth];
    }
    const char* types[] = {"float", "uint", "int", "depth"};
    const char* dimensions[] = {"2d", "array", "3d"};
    uint32_t count = Depth(format) ? 2 : 3;
    for (uint32_t dimension = 0; dimension < count; ++dimension) {
        char name[64];
        std::snprintf(name, sizeof(name), "blit_%s_%s", types[BlitType(format)], dimensions[dimension]);
        MTLRenderPipelineDescriptor* pipeline = [MTLRenderPipelineDescriptor new];
        pipeline.vertexFunction = [state.blitLibrary newFunctionWithName:@"blit_vertex"];
        pipeline.fragmentFunction = [state.blitLibrary newFunctionWithName:[[NSString alloc] initWithUTF8String:name]];
        if (Depth(format)) {
            pipeline.depthAttachmentPixelFormat = PixelFormat(format);
            if (format == Format::D24UnormS8Uint) pipeline.stencilAttachmentPixelFormat = PixelFormat(format);
        } else {
            pipeline.colorAttachments[0].pixelFormat = PixelFormat(format);
        }
        pipelines[dimension] = [state.device newRenderPipelineStateWithDescriptor:pipeline error:&error];
        Require(pipelines[dimension] != nil, "Cannot create blit pipeline: %s", error.localizedDescription.UTF8String);
    }
}

id<MTLFunction> Function(const std::vector<uint8_t>& bytes, const ConstantRaw& constants, metal::ShaderInfo& info) {
    char hash[17];
    std::snprintf(hash, sizeof(hash), "%016llx", (unsigned long long)metal::ShaderHash(bytes));
    const char* directory = std::getenv("EVK_METAL_SHADER_DIR");
    std::string stem = std::string(directory ? directory : ".cache/metal/shaders") + "/" + hash;
    std::ifstream metadata(stem + ".info", std::ios::binary);
    Require(bool(metadata.read(reinterpret_cast<char*>(&info), sizeof(info))), "Missing shader metadata: %s.info", stem.c_str());
    Require(info.magic == 0x4D534C31 && info.version == 6 && info.hash == metal::ShaderHash(bytes), "Invalid shader metadata: %s", stem.c_str());
    NSError* error = nil;
    NSURL* url = [NSURL fileURLWithPath:[[NSString alloc] initWithUTF8String:(stem + ".metallib").c_str()]];
    id<MTLLibrary> library = nil;
    if (info.staticConstants) {
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_1;
        options.fastMathEnabled = YES;
        options.preserveInvariance = YES;
        NSMutableDictionary* macros = [NSMutableDictionary new];
        for (uint32_t index = 0; index < 32; ++index) {
            if (!(info.staticConstants & (1u << index))) continue;
            uint32_t bits = info.constantDefaults[index];
            if (index < constants.count) std::memcpy(&bits, constants.data + index * 4, sizeof(bits));
            NSNumber* value = info.constantTypes[index] == metal::ConstantType::Int ? @(int32_t(bits)) : @(bits);
            if (info.constantTypes[index] == metal::ConstantType::Float) {
                float number;
                std::memcpy(&number, &bits, sizeof(number));
                value = @(number);
            }
            macros[[NSString stringWithFormat:@"SPIRV_CROSS_CONSTANT_ID_%u", index]] = value;
        }
        options.preprocessorMacros = macros;
        NSString* sourcePath = [[NSString alloc] initWithUTF8String:(stem + ".metal").c_str()];
        NSString* source = [NSString stringWithContentsOfFile:sourcePath encoding:NSUTF8StringEncoding error:&error];
        Require(source != nil, "Cannot read array-specialized shader %s: %s", stem.c_str(), error.localizedDescription.UTF8String);
        library = [S().device newLibraryWithSource:source options:options error:&error];
    } else {
        library = [S().device newLibraryWithURL:url error:&error];
    }
    Require(library != nil, "Cannot load shader %s: %s", stem.c_str(), error.localizedDescription.UTF8String);
    MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
    for (uint32_t index = 0; index < 32; ++index) {
        if (info.staticConstants & (1u << index)) continue;
        MTLDataType type = MTLDataTypeNone;
        switch (info.constantTypes[index]) {
            case metal::ConstantType::Bool: type = MTLDataTypeBool; break;
            case metal::ConstantType::Int: type = MTLDataTypeInt; break;
            case metal::ConstantType::UInt: type = MTLDataTypeUInt; break;
            case metal::ConstantType::Float: type = MTLDataTypeFloat; break;
            default: continue;
        }
        uint32_t bits = info.constantDefaults[index];
        if (index < constants.count) std::memcpy(&bits, constants.data + index * 4, sizeof(bits));
        bool boolean = bits != 0;
        const void* value = type == MTLDataTypeBool ? static_cast<const void*>(&boolean) : static_cast<const void*>(&bits);
        [values setConstantValue:value type:type atIndex:index];
    }
    id<MTLFunction> function = [library newFunctionWithName:@"main0" constantValues:values error:&error];
    Require(function != nil, "Cannot specialize shader %s: %s", stem.c_str(), error.localizedDescription.UTF8String);
    return function;
}
}

void Resource::decRef() {
    Require(refCount > 0, "Resource reference underflow");
    if (--refCount) return;
    Require(S().deletions.size() < S().deletions.capacity(), "Resource retirement capacity exceeded");
    S().deletions.push_back({S().nextSubmission, this});
}
RID ResourceRef::GetRID() const { Require(res && res->resourceid >= 0, "Resource has no bindless descriptor"); return res->resourceid; }
const BufferDesc& GetDesc(const Buffer& buffer) { return B(buffer).desc; }
const ImageDesc& GetDesc(const Image& image) { return I(image).desc; }
void* Buffer::GetPtr() {
    Require(GetDesc(*this).memoryType != MemoryType::GPU, "Buffer is not CPU-visible");
    return B(*this).buffer.contents;
}
uint64_t Buffer::GetReference() { return B(*this).buffer.gpuAddress; }

Buffer CreateBuffer(const BufferDesc& desc) {
    @autoreleasepool {
        Require(desc.size > 0 && desc.size <= S().device.maxBufferLength, "Invalid buffer size");
        auto* buffer = new MetalBuffer;
        buffer->desc = desc;
        auto mode = desc.memoryType == MemoryType::GPU ? MTLResourceStorageModePrivate : MTLResourceStorageModeShared;
        buffer->buffer = [S().device newBufferWithLength:desc.size options:mode];
        Require(buffer->buffer != nil, "Cannot allocate buffer %s", desc.name.c_str());
        buffer->buffer.label = [[NSString alloc] initWithUTF8String:desc.name.c_str()];
        AddResource(buffer->buffer, Has(desc.usage, BufferUsage::Storage));
        if (Has(desc.usage, BufferUsage::Storage)) {
            buffer->resourceid = S().buffers.Allocate();
            [S().arguments setBuffer:buffer->buffer offset:0 atIndex:metal::STORAGE_ID + buffer->resourceid];
        }
        return Buffer(buffer);
    }
}
void WriteBuffer(Buffer& buffer, void* bytes, uint64_t size, uint64_t offset) {
    Require(offset <= GetDesc(buffer).size && size <= GetDesc(buffer).size - offset, "Buffer write is out of bounds");
    std::memcpy(static_cast<uint8_t*>(buffer.GetPtr()) + offset, bytes, size);
}
void ReadBuffer(Buffer& buffer, void* bytes, uint64_t size, uint64_t offset) {
    Require(offset <= GetDesc(buffer).size && size <= GetDesc(buffer).size - offset, "Buffer read is out of bounds");
    std::memcpy(bytes, static_cast<uint8_t*>(buffer.GetPtr()) + offset, size);
}

Image CreateImage(const ImageDesc& desc) {
    @autoreleasepool {
        Require(desc.extent.width && desc.extent.height && desc.extent.depth && desc.mipCount && desc.layerCount, "Invalid texture extent");
        auto* image = new MetalImage;
        image->desc = desc;
        MTLTextureDescriptor* texture = [MTLTextureDescriptor new];
        texture.pixelFormat = PixelFormat(desc.format);
        texture.width = desc.extent.width; texture.height = desc.extent.height; texture.depth = desc.extent.depth;
        texture.mipmapLevelCount = desc.mipCount;
        texture.sampleCount = uint32_t(desc.sampleCount);
        texture.arrayLength = desc.layerCount;
        texture.textureType = desc.layerCount > 1 ? MTLTextureType2DArray : MTLTextureType2D;
        if (desc.extent.depth > 1) texture.textureType = MTLTextureType3D;
        if (desc.isCube) {
            Require(desc.layerCount % 6 == 0, "Cube textures need six layers per cube");
            texture.arrayLength = desc.layerCount / 6;
            texture.textureType = texture.arrayLength > 1 ? MTLTextureTypeCubeArray : MTLTextureTypeCube;
        }
        if (desc.sampleCount != SampleCount::One) {
            Require(desc.layerCount == 1 && desc.mipCount == 1, "Multisample arrays are not implemented");
            texture.textureType = MTLTextureType2DMultisample;
        }
        texture.storageMode = MTLStorageModePrivate;
        texture.usage = MTLTextureUsageRenderTarget;
        if (Has(desc.usage, ImageUsage::Sampled) || Has(desc.usage, ImageUsage::TransferSrc)) texture.usage |= MTLTextureUsageShaderRead;
        if (Has(desc.usage, ImageUsage::Storage)) texture.usage |= MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        if (desc.isCube && Has(desc.usage, ImageUsage::TransferSrc)) texture.usage |= MTLTextureUsagePixelFormatView;
        image->texture = [S().device newTextureWithDescriptor:texture];
        Require(image->texture != nil, "Cannot allocate texture %s", desc.name.c_str());
        image->texture.label = [[NSString alloc] initWithUTF8String:desc.name.c_str()];
        image->blitTexture = image->texture;
        if (desc.isCube && Has(desc.usage, ImageUsage::TransferSrc)) {
            image->blitTexture = [image->texture newTextureViewWithPixelFormat:texture.pixelFormat textureType:MTLTextureType2DArray
                levels:NSMakeRange(0, desc.mipCount) slices:NSMakeRange(0, desc.layerCount)];
            Require(image->blitTexture != nil, "Cannot create cube blit view");
        }
        if (Has(desc.usage, ImageUsage::TransferDst) && desc.sampleCount == SampleCount::One) PrepareBlitPipelines(desc.format);
        AddResource(image->texture, Has(desc.usage, ImageUsage::Storage));
        if (Has(desc.usage, ImageUsage::Sampled) || Has(desc.usage, ImageUsage::Storage)) {
            image->resourceid = S().images.Allocate();
            [S().arguments setTexture:image->texture atIndex:metal::TEXTURE_ID + image->resourceid];
            [S().arguments setTexture:image->texture atIndex:metal::IMAGE_ID + image->resourceid];
            [S().arguments setSamplerState:desc.filter == Filter::Nearest ? S().nearest : S().linear atIndex:metal::SAMPLER_ID + image->resourceid];
        }
        return Image(image);
    }
}

Pipeline CreatePipeline(const PipelineDesc& desc) {
    @autoreleasepool {
        auto* pipeline = new MetalPipeline;
        pipeline->desc = desc;
        metal::ShaderInfo info;
        NSError* error = nil;
        if (!desc.CS.empty()) {
            id<MTLFunction> function = Function(desc.CS, desc.constants, info);
            auto groupSize = [&](uint32_t size, uint32_t constant) {
                if (constant < desc.constants.count) std::memcpy(&size, desc.constants.data + constant * 4, sizeof(size));
                Require(size > 0, "Compute workgroup size must be positive");
                return size;
            };
            pipeline->group = MTLSizeMake(groupSize(info.groupX, info.groupConstantX),
                groupSize(info.groupY, info.groupConstantY), groupSize(info.groupZ, info.groupConstantZ));
            auto limit = S().device.maxThreadsPerThreadgroup;
            Require(pipeline->group.width <= limit.width && pipeline->group.height <= limit.height
                && pipeline->group.depth <= limit.depth, "Compute workgroup dimensions exceed device limits");
            MTLComputePipelineDescriptor* compute = [MTLComputePipelineDescriptor new];
            compute.computeFunction = function;
            compute.maxTotalThreadsPerThreadgroup = pipeline->group.width * pipeline->group.height * pipeline->group.depth;
            pipeline->compute = [S().device newComputePipelineStateWithDescriptor:compute options:MTLPipelineOptionNone reflection:nil error:&error];
            Require(pipeline->compute != nil, "Cannot create compute pipeline %s: %s", desc.name.c_str(), error.localizedDescription.UTF8String);
            Require(compute.maxTotalThreadsPerThreadgroup <= pipeline->compute.maxTotalThreadsPerThreadgroup,
                "Compute workgroup size exceeds pipeline limits");
            return Pipeline(pipeline);
        }
        MTLRenderPipelineDescriptor* render = [MTLRenderPipelineDescriptor new];
        render.vertexFunction = Function(desc.VS, desc.constants, info);
        pipeline->usesDrawID = info.usesDrawID;
        Require(!pipeline->usesDrawID || metal::VERTEX_BUFFER + desc.bindings.size() <= metal::DRAW_ID_BUFFER,
                "Vertex bindings overlap the shader draw-ID buffer slot");
        if (!desc.FS.empty()) render.fragmentFunction = Function(desc.FS, desc.constants, info);
        render.rasterSampleCount = uint32_t(desc.sampleCount);
        MTLVertexDescriptor* vertices = [MTLVertexDescriptor new];
        uint32_t attribute = 0;
        for (size_t binding = 0; binding < desc.bindings.size(); ++binding) {
            uint32_t offset = 0;
            for (Format format : desc.bindings[binding]) {
                vertices.attributes[attribute].format = VertexFormat(format);
                vertices.attributes[attribute].offset = offset;
                vertices.attributes[attribute].bufferIndex = metal::VERTEX_BUFFER + binding;
                offset += PixelBytes(format); ++attribute;
            }
            vertices.layouts[metal::VERTEX_BUFFER + binding].stride = offset;
            vertices.layouts[metal::VERTEX_BUFFER + binding].stepFunction = MTLVertexStepFunctionPerVertex;
        }
        if (attribute) render.vertexDescriptor = vertices;
        uint32_t color = 0;
        for (size_t index = 0; index < desc.attachments.size(); ++index) {
            Format format = desc.attachments[index];
            if (Depth(format)) {
                render.depthAttachmentPixelFormat = PixelFormat(format);
                if (format == Format::D24UnormS8Uint) render.stencilAttachmentPixelFormat = PixelFormat(format);
                continue;
            }
            auto attachment = render.colorAttachments[color++];
            attachment.pixelFormat = PixelFormat(format);
            Blend blend = index < desc.blends.size() ? desc.blends[index] : Blend::Disabled;
            attachment.blendingEnabled = blend != Blend::Disabled;
            attachment.sourceRGBBlendFactor = blend == Blend::Alpha ? MTLBlendFactorSourceAlpha : MTLBlendFactorOne;
            attachment.destinationRGBBlendFactor = blend == Blend::Alpha ? MTLBlendFactorOneMinusSourceAlpha : MTLBlendFactorOne;
            attachment.sourceAlphaBlendFactor = blend == Blend::Alpha ? MTLBlendFactorZero : MTLBlendFactorOne;
            attachment.destinationAlphaBlendFactor = blend == Blend::Alpha ? MTLBlendFactorOne : MTLBlendFactorZero;
        }
        pipeline->render = [S().device newRenderPipelineStateWithDescriptor:render error:&error];
        Require(pipeline->render != nil, "Cannot create graphics pipeline %s: %s", desc.name.c_str(), error.localizedDescription.UTF8String);
        MTLDepthStencilDescriptor* depth = [MTLDepthStencilDescriptor new];
        depth.depthCompareFunction = desc.depthTest ? MTLCompareFunction(uint32_t(desc.depthOp)) : MTLCompareFunctionAlways;
        depth.depthWriteEnabled = desc.depthWrite;
        pipeline->depth = [S().device newDepthStencilStateWithDescriptor:depth];
        return Pipeline(pipeline);
    }
}

bool InitializeEVK(const EvkDesc& desc) {
    @autoreleasepool {
        Require(G == nullptr, "Backend already initialized");
        G = new State;
        auto& state = S();
        state.device = MTLCreateSystemDefaultDevice();
        Require(state.device != nil && state.device.argumentBuffersSupport == MTLArgumentBuffersTier2, "A tier-2 Metal device is required");
        state.queue = [state.device newCommandQueueWithMaxCommandBufferCount:COMMAND_COUNT];
        NSError* error = nil;
        auto indirectLibrary = [state.device newLibraryWithSource:[[NSString alloc] initWithUTF8String:metal::INDIRECT_COUNT_SHADER] options:nil error:&error];
        Require(indirectLibrary != nil, "Cannot compile indirect-count shader: %s", error.localizedDescription.UTF8String);
        state.indirectCount = [state.device newComputePipelineStateWithFunction:[indirectLibrary newFunctionWithName:@"filter_draw_count"] error:&error];
        Require(state.indirectCount != nil, "Cannot create indirect-count pipeline: %s", error.localizedDescription.UTF8String);
        state.features.raytracing = state.device.supportsRaytracing;
        state.features.coopmat = [state.device supportsFamily:MTLGPUFamilyApple7];
        state.features.maxFramebufferSampleCount = GetSupportedSampleCount(SampleCount::SixtyFour);
        state.buffers.Initialize(std::min(desc.bindless.storageBufferCount, metal::BUFFER_COUNT));
        state.images.Initialize(std::min(desc.bindless.imageCount, metal::IMAGE_COUNT));
        state.tlas.Initialize(std::min(desc.bindless.tlasCount, metal::TLAS_COUNT));
        state.reads.reserve(metal::BUFFER_COUNT + metal::IMAGE_COUNT + metal::TLAS_COUNT + 4096);
        state.writes.reserve(metal::BUFFER_COUNT + metal::IMAGE_COUNT + metal::TLAS_COUNT + 4096);
        state.deletions.reserve(2 * (metal::BUFFER_COUNT + metal::IMAGE_COUNT + metal::TLAS_COUNT) + 4096);
        state.timestamps.reserve(TIMESTAMP_COUNT);
        NSMutableArray<MTLArgumentDescriptor*>* arguments = [NSMutableArray new];
        for (uint32_t binding = 0; binding < 5; ++binding) {
            MTLArgumentDescriptor* argument = [MTLArgumentDescriptor new];
            argument.index = binding == 0 ? metal::STORAGE_ID : binding == 1 ? metal::TEXTURE_ID : binding == 2 ? metal::SAMPLER_ID : binding == 3 ? metal::IMAGE_ID : metal::TLAS_ID;
            argument.arrayLength = binding == 0 ? metal::BUFFER_COUNT : binding == 4 ? metal::TLAS_COUNT : metal::IMAGE_COUNT;
            argument.dataType = binding == 0 ? MTLDataTypePointer : binding == 2 ? MTLDataTypeSampler : binding == 4 ? MTLDataTypeInstanceAccelerationStructure : MTLDataTypeTexture;
            argument.textureType = MTLTextureType2D;
            argument.access = binding == 0 || binding == 3 ? MTLBindingAccessReadWrite : MTLBindingAccessReadOnly;
            [arguments addObject:argument];
        }
        state.arguments = [state.device newArgumentEncoderWithArguments:arguments];
        Require(state.arguments != nil, "Cannot create bindless encoder");
        state.argumentBuffer = [state.device newBufferWithLength:state.arguments.encodedLength options:MTLResourceStorageModeShared];
        Require(state.argumentBuffer != nil, "Cannot allocate bindless table");
        [state.arguments setArgumentBuffer:state.argumentBuffer offset:0];
        MTLSamplerDescriptor* sampler = [MTLSamplerDescriptor new];
        sampler.supportArgumentBuffers = YES;
        sampler.sAddressMode = sampler.tAddressMode = sampler.rAddressMode = MTLSamplerAddressModeClampToEdge;
        sampler.minFilter = sampler.magFilter = MTLSamplerMinMagFilterNearest;
        sampler.mipFilter = MTLSamplerMipFilterNearest;
        state.nearest = [state.device newSamplerStateWithDescriptor:sampler];
        sampler.minFilter = sampler.magFilter = MTLSamplerMinMagFilterLinear;
        state.linear = [state.device newSamplerStateWithDescriptor:sampler];
        id<MTLCounterSet> counterSet = nil;
        for (id<MTLCounterSet> set in state.device.counterSets) if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) counterSet = set;
        state.features.timestamps = desc.enableTimestamps && counterSet != nil && [state.device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];
        for (uint32_t samples = 1; samples <= 8; samples *= 2) if ([state.device supportsTextureSampleCount:samples]) state.features.maxFramebufferSampleCount = SampleCount(samples);
        for (auto& cmd : state.commands) {
            cmd.api._internal = &cmd;
            cmd.drawableImage = Image(new MetalImage);
            cmd.renderPass = [MTLRenderPassDescriptor new];
            cmd.computePass = [MTLComputePassDescriptor new];
            cmd.blitPass = [MTLBlitPassDescriptor new];
            cmd.accelerationPass = [MTLAccelerationStructurePassDescriptor new];
            cmd.staging = CreateBuffer({.name = "Metal command staging", .size = STAGING_BYTES, .usage = BufferUsage::TransferSrc, .memoryType = MemoryType::CPU_TO_GPU});
            if (state.features.timestamps) {
                MTLCounterSampleBufferDescriptor* counters = [MTLCounterSampleBufferDescriptor new];
                counters.counterSet = counterSet; counters.sampleCount = SAMPLE_COUNT; counters.storageMode = MTLStorageModeShared;
                NSError* error = nil;
                cmd.counters = [state.device newCounterSampleBufferWithDescriptor:counters error:&error];
                Require(cmd.counters != nil, "Cannot create timestamp buffer: %s", error.localizedDescription.UTF8String);
            }
        }
        std::printf("[evk] Backend Metal | %s | stage_boundary_timestamps=%d | ray_api=%d\n", state.device.name.UTF8String, state.features.timestamps, state.features.raytracing);
        return true;
    }
}
bool InitializeSwapchain(void* nativeWindow) {
    @autoreleasepool {
        NSWindow* window = (__bridge NSWindow*)nativeWindow;
        Require(window != nil, "Missing Cocoa window");
        S().view = window.contentView;
        S().layer = [CAMetalLayer new];
        S().layer.device = S().device;
        S().layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        S().layer.framebufferOnly = YES;
        S().layer.maximumDrawableCount = 3;
        S().view.wantsLayer = YES;
        S().view.layer = S().layer;
        ResizeLayer();
        return true;
    }
}
const Features& GetFeatures() { return S().features; }
SampleCount GetSupportedSampleCount(SampleCount requested) {
    uint32_t count = uint32_t(requested);
    while (count > 1 && ![S().device supportsTextureSampleCount:count]) count /= 2;
    return SampleCount(count);
}
Extent GetSwapchainExtent() { ResizeLayer(); return {uint32_t(S().layer.drawableSize.width), uint32_t(S().layer.drawableSize.height)}; }
void RequestSwapchainRecreate() { ResizeLayer(); }
MemoryBudget GetMemoryBudget() {
    MemoryBudget budget = {};
    auto& heap = budget.heaps[0];
    heap.allocationBytes = S().liveBytes + S().argumentBuffer.allocatedSize;
    heap.allocationCount = S().allocations + 1;
    heap.blockBytes = heap.usage = S().device.currentAllocatedSize;
    heap.blockCount = heap.allocationCount;
    heap.budget = S().device.recommendedMaxWorkingSetSize;
    return budget;
}
const std::vector<TimestampEntry>& CmdTimestamps() { return S().timestamps; }

Cmd& CmdBegin(Queue) {
    @autoreleasepool {
        for (auto& cmd : S().commands) if (cmd.pending && cmd.buffer.status == MTLCommandBufferStatusCompleted) Complete(cmd);
        for (auto& cmd : S().commands) {
            if (cmd.pending || cmd.recording) continue;
            cmd.buffer = [S().queue commandBuffer];
            Require(cmd.buffer != nil, "Cannot create command buffer");
            cmd.recording = true;
            cmd.stagingOffset = 0; cmd.scratchBytes = 0; cmd.markerCount = 0; cmd.samples = 0; cmd.push.fill(0);
            return cmd.api;
        }
        auto oldest = std::min_element(S().commands.begin(), S().commands.end(), [](const Command& a, const Command& b) { return a.submission < b.submission; });
        Require(oldest->pending, "All command buffers are still recording");
        CmdWait(oldest->submission);
        return CmdBegin();
    }
}
bool CmdDone(uint64_t submission) {
    if (submission <= S().completed) return true;
    for (auto& cmd : S().commands) {
        if (!cmd.pending || cmd.submission != submission) continue;
        if (cmd.buffer.status != MTLCommandBufferStatusCompleted) return false;
        Complete(cmd); return true;
    }
    return false;
}
void CmdWait(uint64_t submission) {
    if (submission <= S().completed) return;
    @autoreleasepool {
        for (auto& cmd : S().commands) {
            if (!cmd.pending || cmd.submission != submission) continue;
            [cmd.buffer waitUntilCompleted]; Complete(cmd); return;
        }
        Require(false, "Unknown submission %llu", (unsigned long long)submission);
    }
}
uint64_t Cmd::submit() {
    @autoreleasepool {
        auto& cmd = C(*this);
        Require(cmd.recording && !cmd.render, "Submission contains an unfinished render pass");
        EndEncoder(cmd);
        if (cmd.drawable) [cmd.buffer presentDrawable:cmd.drawable];
        cmd.submission = S().nextSubmission++;
        cmd.pending = true; cmd.recording = false;
        [cmd.buffer commit];
        return cmd.submission;
    }
}
void Cmd::push(void* bytes, uint32_t size, uint32_t offset) {
    Require(offset <= C(*this).push.size() && size <= C(*this).push.size() - offset, "Push constants exceed 256 bytes");
    std::memcpy(C(*this).push.data() + offset, bytes, size);
}
void Cmd::bind(Pipeline pipeline) { C(*this).pipeline = pipeline; }
void Cmd::dispatch(uint32_t x, uint32_t y, uint32_t z) {
    auto& cmd = C(*this); Compute(cmd);
    auto& pipeline = P(cmd.pipeline);
    Require(pipeline.compute != nil, "Dispatch requires a compute pipeline");
    [cmd.compute setComputePipelineState:pipeline.compute]; Push(cmd);
    [cmd.compute dispatchThreadgroups:MTLSizeMake(x, y, z) threadsPerThreadgroup:pipeline.group];
}
void Cmd::restoreBindings() { if (C(*this).render) Residency(C(*this).render); if (C(*this).compute) Residency(C(*this).compute); }
void Cmd::barrier(Image&, ImageLayout, ImageLayout, uint32_t, uint32_t, uint32_t, uint32_t) { Require(!C(*this).render, "Image transition inside a render pass"); EndEncoder(C(*this)); }
void Cmd::barrier() { Require(!C(*this).render, "Global barrier inside a render pass"); EndEncoder(C(*this)); }
void Cmd::computeBarrier() { if (C(*this).compute) [C(*this).compute memoryBarrierWithScope:MTLBarrierScopeBuffers | MTLBarrierScopeTextures]; }
void Cmd::copy(Buffer& src, Buffer& dst, uint64_t size, uint64_t srcOffset, uint64_t dstOffset) {
    Require(srcOffset <= GetDesc(src).size && size <= GetDesc(src).size - srcOffset && dstOffset <= GetDesc(dst).size && size <= GetDesc(dst).size - dstOffset, "Buffer copy is out of bounds");
    auto& cmd = C(*this); Blit(cmd);
    [cmd.blit copyFromBuffer:B(src).buffer sourceOffset:srcOffset toBuffer:B(dst).buffer destinationOffset:dstOffset size:size];
}
void Cmd::copy(void* src, Buffer& dst, uint64_t size, uint64_t dstOffset) {
    auto& cmd = C(*this); uint64_t offset = StageUpload(cmd, src, size); copy(cmd.staging, dst, size, offset, dstOffset);
}
void Cmd::update(Buffer& dst, uint64_t offset, uint64_t size, void* src) { copy(src, dst, size, offset); }
void Cmd::fill(Buffer dst, uint32_t value, uint64_t size, uint64_t offset) {
    auto& cmd = C(*this);
    Require(offset <= GetDesc(dst).size && size <= GetDesc(dst).size - offset && size % 4 == 0, "Buffer fill is out of bounds or unaligned");
    if ((value & 255) * 0x01010101u == value) {
        Blit(cmd); [cmd.blit fillBuffer:B(dst).buffer range:NSMakeRange(offset, size) value:uint8_t(value)]; return;
    }
    uint64_t stage = (cmd.stagingOffset + 255) & ~uint64_t(255);
    Require(stage <= STAGING_BYTES && size <= STAGING_BYTES - stage, "Staging buffer fill capacity exceeded");
    std::fill_n(reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(cmd.staging.GetPtr()) + stage), size / 4, value);
    cmd.stagingOffset = stage + size; copy(cmd.staging, dst, size, stage, offset);
}

void Cmd::copy(Buffer& src, Image& dst, uint32_t mip, uint32_t layer) {
    const auto& desc = GetDesc(dst);
    uint32_t width = std::max(1u, desc.extent.width >> mip), height = std::max(1u, desc.extent.height >> mip), depth = std::max(1u, desc.extent.depth >> mip);
    Require(mip < desc.mipCount && layer < desc.layerCount, "Texture upload subresource is out of bounds");
    auto& cmd = C(*this); Blit(cmd);
    [cmd.blit copyFromBuffer:B(src).buffer sourceOffset:0 sourceBytesPerRow:width * PixelBytes(desc.format) sourceBytesPerImage:width * height * PixelBytes(desc.format)
        sourceSize:MTLSizeMake(width, height, depth) toTexture:I(dst).texture destinationSlice:layer destinationLevel:mip destinationOrigin:MTLOriginMake(0, 0, 0)];
}
void Cmd::copy(void* src, Image& dst, uint64_t size, uint32_t mip, uint32_t layer) {
    const auto& desc = GetDesc(dst);
    uint32_t width = std::max(1u, desc.extent.width >> mip), height = std::max(1u, desc.extent.height >> mip), depth = std::max(1u, desc.extent.depth >> mip);
    Require(mip < desc.mipCount && layer < desc.layerCount && size == uint64_t(width) * height * depth * PixelBytes(desc.format), "Invalid texture upload size");
    auto& cmd = C(*this); uint64_t offset = StageUpload(cmd, src, size); Blit(cmd);
    [cmd.blit copyFromBuffer:B(cmd.staging).buffer sourceOffset:offset sourceBytesPerRow:width * PixelBytes(desc.format) sourceBytesPerImage:width * height * PixelBytes(desc.format)
        sourceSize:MTLSizeMake(width, height, depth) toTexture:I(dst).texture destinationSlice:layer destinationLevel:mip destinationOrigin:MTLOriginMake(0, 0, 0)];
}
void Cmd::copy(Image& src, Buffer& dst, uint32_t mip, uint32_t layer) {
    const auto& desc = GetDesc(src);
    Require(mip < desc.mipCount && layer < desc.layerCount, "Readback subresource is out of bounds");
    uint32_t width = std::max(1u, desc.extent.width >> mip), height = std::max(1u, desc.extent.height >> mip), depth = std::max(1u, desc.extent.depth >> mip);
    Require(GetDesc(dst).size >= uint64_t(width) * height * depth * PixelBytes(desc.format), "Readback buffer is too small");
    auto& cmd = C(*this); Blit(cmd);
    [cmd.blit copyFromTexture:I(src).texture sourceSlice:layer sourceLevel:mip sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(width, height, depth)
        toBuffer:B(dst).buffer destinationOffset:0 destinationBytesPerRow:width * PixelBytes(desc.format) destinationBytesPerImage:width * height * PixelBytes(desc.format)];
}
void Cmd::copy(Image& src, Image& dst, uint32_t srcMip, uint32_t srcLayer, uint32_t dstMip, uint32_t dstLayer, uint32_t layers) {
    const auto& desc = GetDesc(src);
    auto& cmd = C(*this); Blit(cmd);
    MTLSize size = MTLSizeMake(std::max(1u, desc.extent.width >> srcMip), std::max(1u, desc.extent.height >> srcMip), std::max(1u, desc.extent.depth >> srcMip));
    for (uint32_t layer = 0; layer < layers; ++layer) [cmd.blit copyFromTexture:I(src).texture sourceSlice:srcLayer + layer sourceLevel:srcMip sourceOrigin:MTLOriginMake(0, 0, 0)
        sourceSize:size toTexture:I(dst).texture destinationSlice:dstLayer + layer destinationLevel:dstMip destinationOrigin:MTLOriginMake(0, 0, 0)];
}
void Cmd::copy(Buffer& src, Image& dst, const std::vector<ImageRegion>& regions) {
    auto& cmd = C(*this); Blit(cmd);
    uint64_t offset = 0;
    for (const auto& region : regions) {
        uint32_t bytes = PixelBytes(GetDesc(dst).format);
        [cmd.blit copyFromBuffer:B(src).buffer sourceOffset:offset sourceBytesPerRow:region.width * bytes sourceBytesPerImage:region.width * region.height * bytes
            sourceSize:MTLSizeMake(region.width, region.height, region.depth) toTexture:I(dst).texture destinationSlice:region.layer destinationLevel:region.mip destinationOrigin:MTLOriginMake(region.x, region.y, region.z)];
        offset += uint64_t(region.width) * region.height * region.depth * bytes;
    }
}
namespace {
void RenderEncoder(Command& cmd) {
    auto pass = cmd.renderPass;
    if (cmd.counters) {
        uint32_t first = EncoderSamples(cmd);
        auto sample = pass.sampleBufferAttachments[0];
        sample.sampleBuffer = cmd.counters;
        sample.startOfVertexSampleIndex = first;
        sample.endOfVertexSampleIndex = first + 1;
        sample.startOfFragmentSampleIndex = first + 2;
        sample.endOfFragmentSampleIndex = first + 3;
    }
    cmd.render = [cmd.buffer renderCommandEncoderWithDescriptor:pass];
    Require(cmd.render != nil, "Cannot create render encoder");
    [cmd.render setViewport:cmd.viewport];
    [cmd.render setScissorRect:cmd.scissor];
    [cmd.render setVertexBuffer:S().argumentBuffer offset:0 atIndex:metal::ARGUMENT_BUFFER];
    [cmd.render setFragmentBuffer:S().argumentBuffer offset:0 atIndex:metal::ARGUMENT_BUFFER];
    Residency(cmd.render);
}

void BeginRender(Cmd& api, Image* attachments, ClearValue* clears, int count, Image* resolves, bool loadDepth, uint32_t mip, uint32_t layer, bool loadColor = false) {
    auto& cmd = C(api);
    Require(!cmd.render && count > 0 && count <= MAX_ATTACHMENTS_COUNT, "Invalid render pass");
    EndEncoder(cmd);
    @autoreleasepool {
        auto pass = cmd.renderPass;
        for (uint32_t index = 0; index < MAX_ATTACHMENTS_COUNT; ++index) { pass.colorAttachments[index].texture = nil; pass.colorAttachments[index].resolveTexture = nil; }
        pass.depthAttachment.texture = nil;
        pass.stencilAttachment.texture = nil;
        uint32_t color = 0;
        for (int index = 0; index < count; ++index) {
            const auto& desc = GetDesc(attachments[index]);
            auto action = clears ? MTLLoadActionClear : MTLLoadActionDontCare;
            auto store = Has(desc.usage, ImageUsage::Transient) ? MTLStoreActionDontCare : MTLStoreActionStore;
            if (Depth(desc.format)) {
                auto attachment = pass.depthAttachment;
                attachment.texture = I(attachments[index]).texture;
                attachment.level = mip; attachment.slice = layer; attachment.depthPlane = 0;
                attachment.loadAction = loadDepth ? MTLLoadActionLoad : action;
                cmd.depthStore = store;
                attachment.storeAction = MTLStoreActionUnknown;
                attachment.clearDepth = clears ? clears[index].depthStencil.depth : 1.0;
                if (desc.format == Format::D24UnormS8Uint) {
                    pass.stencilAttachment.texture = attachment.texture;
                    pass.stencilAttachment.level = mip; pass.stencilAttachment.slice = layer; pass.stencilAttachment.depthPlane = 0;
                    pass.stencilAttachment.loadAction = attachment.loadAction;
                    cmd.stencilStore = store;
                    pass.stencilAttachment.storeAction = MTLStoreActionUnknown;
                    pass.stencilAttachment.clearStencil = clears ? clears[index].depthStencil.stencil : 0;
                }
                continue;
            }
            cmd.colorStores[color] = store;
            auto attachment = pass.colorAttachments[color++];
            attachment.texture = I(attachments[index]).texture;
            attachment.level = mip;
            attachment.slice = desc.extent.depth > 1 ? 0 : layer;
            attachment.depthPlane = desc.extent.depth > 1 ? layer : 0;
            attachment.loadAction = loadColor ? MTLLoadActionLoad : action; attachment.storeAction = MTLStoreActionUnknown;
            if (clears) {
                auto& value = clears[index].color;
                attachment.clearColor = UInt(desc.format) ? MTLClearColorMake(value.uint32[0], value.uint32[1], value.uint32[2], value.uint32[3])
                    : desc.format == Format::RGBA32Sint ? MTLClearColorMake(value.int32[0], value.int32[1], value.int32[2], value.int32[3])
                    : MTLClearColorMake(value.float32[0], value.float32[1], value.float32[2], value.float32[3]);
            }
            if (resolves && resolves[index]) { attachment.resolveTexture = I(resolves[index]).texture; cmd.colorStores[color - 1] = MTLStoreActionMultisampleResolve; }
        }
        auto extent = GetDesc(attachments[0]).extent;
        extent.width = std::max(1u, extent.width >> mip);
        extent.height = std::max(1u, extent.height >> mip);
        cmd.viewport = {0, 0, double(extent.width), double(extent.height), 0, 1};
        cmd.scissor = {0, 0, extent.width, extent.height};
        RenderEncoder(cmd);
    }
}

Extent BlitRegion(const ImageDesc& desc, ImageRegion& region) {
    Require(region.mip >= 0 && uint32_t(region.mip) < desc.mipCount && region.layer >= 0 && uint32_t(region.layer) < desc.layerCount,
        "Blit subresource is out of bounds");
    Extent extent = {std::max(1u, desc.extent.width >> region.mip), std::max(1u, desc.extent.height >> region.mip), std::max(1u, desc.extent.depth >> region.mip)};
    if (region.width == 0) region.width = extent.width;
    if (region.height == 0) region.height = extent.height;
    if (region.depth == 0) region.depth = extent.depth;
    auto valid = [](int offset, int size, uint32_t limit) {
        int64_t end = int64_t(offset) + size;
        return std::min(int64_t(offset), end) >= 0 && std::max(int64_t(offset), end) <= limit;
    };
    Require(valid(region.x, region.width, extent.width) && valid(region.y, region.height, extent.height) && valid(region.z, region.depth, extent.depth),
        "Blit region is out of bounds");
    return extent;
}
}
void Cmd::blit(Image& src, Image& dst, ImageRegion from, ImageRegion to, Filter filter) {
    const auto& source = GetDesc(src);
    const auto& target = GetDesc(dst);
    auto extent = BlitRegion(source, from);
    BlitRegion(target, to);
    Require(source.sampleCount == SampleCount::One && target.sampleCount == SampleCount::One && Has(source.usage, ImageUsage::TransferSrc)
        && Has(target.usage, ImageUsage::TransferDst), "Blit requires single-sample transfer textures");
    Require(BlitType(source.format) == BlitType(target.format), "Blit formats have incompatible numeric types");
    Require(BlitType(source.format) == 0 || filter == Filter::Nearest, "Integer/depth blits require nearest filtering");
    Require(!Depth(source.format) || source.format == target.format, "Depth blit formats must match");
    auto& cmd = C(*this);
    if (source.format == target.format && from.width > 0 && from.height > 0 && from.depth > 0
        && from.width == to.width && from.height == to.height && from.depth == to.depth) {
        Blit(cmd);
        [cmd.blit copyFromTexture:I(src).texture sourceSlice:from.layer sourceLevel:from.mip sourceOrigin:MTLOriginMake(from.x, from.y, from.z)
            sourceSize:MTLSizeMake(from.width, from.height, from.depth) toTexture:I(dst).texture destinationSlice:to.layer destinationLevel:to.mip destinationOrigin:MTLOriginMake(to.x, to.y, to.z)];
        return;
    }
    uint32_t dimension = source.extent.depth > 1 ? 2 : source.layerCount > 1 ? 1 : 0;
    auto pipeline = S().blitPipelines[uint32_t(target.format)][dimension];
    Require(pipeline != nil, "Blit pipeline was not prepared");
    struct alignas(16) Params { float origin[4]; float scale[4]; uint32_t mip; uint32_t layer; } params = {};
    params.scale[0] = float(from.width) / to.width / extent.width;
    params.scale[1] = float(from.height) / to.height / extent.height;
    params.origin[0] = float(from.x) / extent.width - to.x * params.scale[0];
    params.origin[1] = float(from.y) / extent.height - to.y * params.scale[1];
    params.mip = from.mip; params.layer = from.layer;
    int firstPlane = std::min(to.z, to.z + to.depth);
    int lastPlane = std::max(to.z, to.z + to.depth);
    for (int plane = firstPlane; plane < lastPlane; ++plane) {
        params.origin[2] = (from.z + (plane + 0.5f - to.z) * float(from.depth) / to.depth) / extent.depth;
        BeginRender(*this, &dst, nullptr, 1, nullptr, true, to.mip, target.extent.depth > 1 ? plane : to.layer, true);
        [cmd.render setRenderPipelineState:pipeline];
        [cmd.render setDepthStencilState:Depth(target.format) ? S().blitDepth : S().blitColorDepth];
        [cmd.render setScissorRect:MTLScissorRect{NSUInteger(std::min(to.x, to.x + to.width)), NSUInteger(std::min(to.y, to.y + to.height)), NSUInteger(std::abs(to.width)), NSUInteger(std::abs(to.height))}];
        [cmd.render setFragmentTexture:I(src).blitTexture atIndex:0];
        [cmd.render setFragmentSamplerState:filter == Filter::Nearest ? S().nearest : S().linear atIndex:0];
        [cmd.render setFragmentBytes:&params length:sizeof(params) atIndex:0];
        [cmd.render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        endRender();
    }
}
void Cmd::beginRender(Image* attachments, ClearValue* clears, int count, Image* resolves, bool loadDepth) {
    BeginRender(*this, attachments, clears, count, resolves, loadDepth, 0, 0);
}
void Cmd::endRender() { Require(C(*this).render != nil, "No active render pass"); EndEncoder(C(*this)); }
void Cmd::clear(Image image, ClearValue value) {
    const auto& desc = GetDesc(image);
    for (uint32_t mip = 0; mip < desc.mipCount; ++mip) {
        uint32_t slices = desc.extent.depth > 1 ? std::max(1u, desc.extent.depth >> mip) : desc.layerCount;
        for (uint32_t layer = 0; layer < slices; ++layer) {
            BeginRender(*this, &image, &value, 1, nullptr, false, mip, layer);
            endRender();
        }
    }
}
void Cmd::vertex(Buffer& buffer, uint64_t offset) { C(*this).vertices = buffer; C(*this).vertexOffset = offset; }
void Cmd::index(Buffer& buffer, bool half, uint64_t offset) { C(*this).indices = buffer; C(*this).halfIndices = half; C(*this).indexOffset = offset; }
void Cmd::viewport(float x, float y, float width, float height, float near, float far) {
    auto& cmd = C(*this);
    Require(cmd.render != nil, "Viewport requires render pass");
    cmd.viewport = {double(x), double(y + height), double(width), double(-height), double(near), double(far)};
    [cmd.render setViewport:cmd.viewport];
}
void Cmd::scissor(int32_t x, int32_t y, uint32_t width, uint32_t height) {
    Require(C(*this).render != nil && x >= 0 && y >= 0, "Invalid scissor");
    auto& cmd = C(*this);
    cmd.scissor = {NSUInteger(x), NSUInteger(y), width, height};
    [cmd.render setScissorRect:cmd.scissor];
}
void Cmd::lineWidth(float) {}
namespace {
void DrawState(Command& cmd) {
    Require(cmd.render != nil, "Draw requires a render pass");
    auto& pipeline = P(cmd.pipeline);
    Require(pipeline.render != nil, "Draw requires graphics pipeline");
    [cmd.render setRenderPipelineState:pipeline.render];
    [cmd.render setDepthStencilState:pipeline.depth];
    [cmd.render setCullMode:pipeline.desc.cull == Cull::None ? MTLCullModeNone : pipeline.desc.cull == Cull::Front ? MTLCullModeFront : MTLCullModeBack];
    [cmd.render setFrontFacingWinding:pipeline.desc.frontClockwise ? MTLWindingClockwise : MTLWindingCounterClockwise];
    [cmd.render setTriangleFillMode:pipeline.desc.wireframe ? MTLTriangleFillModeLines : MTLTriangleFillModeFill];
    if (cmd.vertices) [cmd.render setVertexBuffer:B(cmd.vertices).buffer offset:cmd.vertexOffset atIndex:metal::VERTEX_BUFFER];
    Push(cmd);
}
MTLPrimitiveType PrimitiveType(Command& cmd) { return P(cmd.pipeline).desc.primitive == Primitive::Triangle ? MTLPrimitiveTypeTriangle : MTLPrimitiveTypeLine; }
void DrawID(Command& cmd, uint32_t index) {
    if (!P(cmd.pipeline).usesDrawID) return;
    [cmd.render setVertexBytes:&index length:sizeof(index) atIndex:metal::DRAW_ID_BUFFER];
}
}
void Cmd::draw(uint32_t vertices, uint32_t instances, uint32_t first, uint32_t baseInstance) {
    auto& cmd = C(*this); DrawState(cmd);
    DrawID(cmd, 0);
    [cmd.render drawPrimitives:PrimitiveType(cmd) vertexStart:first vertexCount:vertices instanceCount:instances baseInstance:baseInstance];
}
void Cmd::drawIndexed(uint32_t indices, uint32_t instances, uint32_t first, int32_t baseVertex, uint32_t baseInstance) {
    auto& cmd = C(*this); DrawState(cmd);
    DrawID(cmd, 0);
    [cmd.render drawIndexedPrimitives:PrimitiveType(cmd) indexCount:indices indexType:cmd.halfIndices ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
        indexBuffer:B(cmd.indices).buffer indexBufferOffset:cmd.indexOffset + first * (cmd.halfIndices ? 2 : 4) instanceCount:instances baseVertex:baseVertex baseInstance:baseInstance];
}
void Cmd::drawIndirect(Buffer& buffer, uint64_t offset, uint32_t count, uint32_t stride) {
    auto& cmd = C(*this); DrawState(cmd);
    for (uint32_t index = 0; index < count; ++index) {
        DrawID(cmd, index);
        [cmd.render drawPrimitives:PrimitiveType(cmd) indirectBuffer:B(buffer).buffer indirectBufferOffset:offset + index * stride];
    }
}
void Cmd::drawIndexedIndirect(Buffer& buffer, uint64_t offset, uint32_t count, uint32_t stride) {
    auto& cmd = C(*this); DrawState(cmd);
    for (uint32_t index = 0; index < count; ++index) {
        DrawID(cmd, index);
        [cmd.render drawIndexedPrimitives:PrimitiveType(cmd) indexType:cmd.halfIndices ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
            indexBuffer:B(cmd.indices).buffer indexBufferOffset:cmd.indexOffset indirectBuffer:B(buffer).buffer indirectBufferOffset:offset + index * stride];
    }
}
namespace {
uint64_t FilterDrawCount(Command& cmd, Buffer& source, uint64_t offset, Buffer& count, uint64_t countOffset, uint32_t maximum, uint32_t stride, uint32_t words) {
    Require(cmd.render != nil, "Counted draw requires a render pass");
    Require(offset % 4 == 0 && countOffset % 4 == 0 && stride % 4 == 0 && stride >= words * 4, "Invalid indirect arguments alignment/stride");
    uint64_t size = maximum ? uint64_t(maximum - 1) * stride + words * 4 : 0;
    Require(offset <= GetDesc(source).size && size <= GetDesc(source).size - offset, "Indirect arguments exceed buffer size");
    Require(countOffset <= GetDesc(count).size && 4 <= GetDesc(count).size - countOffset, "Indirect count exceeds buffer size");
    uint64_t filtered = Scratch(cmd, uint64_t(maximum) * words * 4);
    EndRenderEncoder(cmd, true);
    Compute(cmd);
    struct { uint32_t stride, words, maximum; } params = {stride / 4, words, maximum};
    [cmd.compute setComputePipelineState:S().indirectCount];
    [cmd.compute setBuffer:B(source).buffer offset:offset atIndex:0];
    [cmd.compute setBuffer:B(count).buffer offset:countOffset atIndex:1];
    [cmd.compute setBuffer:B(cmd.staging).buffer offset:filtered atIndex:2];
    [cmd.compute setBytes:&params length:sizeof(params) atIndex:3];
    [cmd.compute dispatchThreads:MTLSizeMake(maximum, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min(NSUInteger(64), S().indirectCount.maxTotalThreadsPerThreadgroup), 1, 1)];
    EndEncoder(cmd);
    for (uint32_t index = 0; index < MAX_ATTACHMENTS_COUNT; ++index) cmd.renderPass.colorAttachments[index].loadAction = MTLLoadActionLoad;
    cmd.renderPass.depthAttachment.loadAction = MTLLoadActionLoad;
    cmd.renderPass.stencilAttachment.loadAction = MTLLoadActionLoad;
    RenderEncoder(cmd);
    return filtered;
}
}
void Cmd::drawIndirectCount(Buffer& buffer, uint64_t offset, Buffer& countBuffer, uint64_t countOffset, uint32_t count, uint32_t stride) {
    if (count == 0) return;
    auto& cmd = C(*this);
    uint64_t filtered = FilterDrawCount(cmd, buffer, offset, countBuffer, countOffset, count, stride, 4);
    drawIndirect(cmd.staging, filtered, count, 16);
}
void Cmd::drawIndexedIndirectCount(Buffer& buffer, uint64_t offset, Buffer& countBuffer, uint64_t countOffset, uint32_t count, uint32_t stride) {
    if (count == 0) return;
    auto& cmd = C(*this);
    uint64_t filtered = FilterDrawCount(cmd, buffer, offset, countBuffer, countOffset, count, stride, 5);
    drawIndexedIndirect(cmd.staging, filtered, count, 20);
}
int Cmd::beginTimestamp(const char* name) {
    auto& cmd = C(*this);
    if (!cmd.counters) return -1;
    Require(!cmd.render, "Timestamp boundaries inside render encoders are unsupported on this GPU");
    EndEncoder(cmd);
    Require(cmd.markerCount < TIMESTAMP_COUNT, "Timestamp marker capacity exceeded");
    uint32_t id = cmd.markerCount++;
    cmd.markers[id] = {name, cmd.samples, cmd.samples};
    return int(id);
}
void Cmd::endTimestamp(int id) {
    if (id < 0) return;
    auto& cmd = C(*this);
    Require(!cmd.render && uint32_t(id) < cmd.markerCount, "Invalid encoder timestamp boundary");
    EndEncoder(cmd); cmd.markers[id].last = cmd.samples;
}
void Cmd::beginPresent() { beginPresent(nullptr, nullptr, 0); }
void Cmd::beginPresent(Image* attachments, ClearValue* clearValues, int count) {
    auto& cmd = C(*this);
    Require(!cmd.drawable && S().layer != nil && count < MAX_ATTACHMENTS_COUNT, "Invalid presentation");
    @autoreleasepool {
        ResizeLayer();
        cmd.drawable = [S().layer nextDrawable];
        Require(cmd.drawable != nil, "Cannot acquire Metal drawable");
        auto& image = I(cmd.drawableImage);
        image.texture = cmd.drawable.texture;
        image.desc = {.extent = {uint32_t(image.texture.width), uint32_t(image.texture.height)}, .format = Format::BGRA8Unorm, .usage = ImageUsage::Attachment};
        std::array<Image, MAX_ATTACHMENTS_COUNT> images;
        std::array<ClearValue, MAX_ATTACHMENTS_COUNT> clears = {
            ClearColor{}, ClearColor{}, ClearColor{}, ClearColor{}, ClearColor{}, ClearColor{}, ClearColor{}, ClearColor{},
        };
        images[0] = cmd.drawableImage; clears[0].color.float32[3] = 1;
        for (int index = 0; index < count; ++index) { images[index + 1] = attachments[index]; if (clearValues) clears[index + 1] = clearValues[index]; }
        beginRender(images.data(), clears.data(), count + 1);
    }
}
void Cmd::endPresent() { endRender(); }
BLAS CreateBLAS(const BLASDesc& geometry) {
    @autoreleasepool {
        Require(S().device.supportsRaytracing, "Device does not support ray tracing");
        auto* blas = new MetalBLAS;
        blas->geometry = geometry;
        MTLAccelerationStructureGeometryDescriptor* desc = nil;
        if (geometry.geometry == GeometryType::Triangles) {
            Require(geometry.stride >= 12 && geometry.stride % 4 == 0 && geometry.vertexCount && geometry.triangleCount
                && GetDesc(geometry.vertices).size >= uint64_t(geometry.stride) * geometry.vertexCount
                && GetDesc(geometry.indices).size >= uint64_t(geometry.triangleCount) * 12, "Invalid triangle BLAS input");
            auto triangles = [MTLAccelerationStructureTriangleGeometryDescriptor new];
            triangles.vertexBuffer = B(geometry.vertices).buffer;
            triangles.vertexFormat = MTLAttributeFormatFloat3;
            triangles.vertexStride = geometry.stride;
            triangles.indexBuffer = B(geometry.indices).buffer;
            triangles.indexType = MTLIndexTypeUInt32;
            triangles.triangleCount = geometry.triangleCount;
            blas->triangleData = CreateBuffer({.name = "BLAS triangle positions", .size = uint64_t(geometry.triangleCount) * 36,
                .usage = BufferUsage::AccelerationStructure, .memoryType = MemoryType::GPU});
            triangles.primitiveDataBuffer = B(blas->triangleData).buffer;
            triangles.primitiveDataStride = triangles.primitiveDataElementSize = 36;
            PrepareTriangleDataPipeline();
            desc = triangles;
            desc.opaque = YES;
        } else {
            Require(geometry.geometry == GeometryType::AABBs && geometry.stride >= sizeof(AABB) && geometry.stride % 4 == 0
                && geometry.aabbsCount && GetDesc(geometry.aabbs).size >= uint64_t(geometry.stride) * geometry.aabbsCount, "Invalid AABB BLAS input");
            auto boxes = [MTLAccelerationStructureBoundingBoxGeometryDescriptor new];
            boxes.boundingBoxBuffer = B(geometry.aabbs).buffer;
            boxes.boundingBoxStride = geometry.stride;
            boxes.boundingBoxCount = geometry.aabbsCount;
            desc = boxes;
            desc.opaque = NO;
        }
        desc.allowDuplicateIntersectionFunctionInvocation = NO;
        blas->desc = [MTLPrimitiveAccelerationStructureDescriptor new];
        blas->desc.geometryDescriptors = @[desc];
        blas->desc.usage = MTLAccelerationStructureUsageRefit;
        auto sizes = [S().device accelerationStructureSizesWithDescriptor:blas->desc];
        blas->scratchBytes = std::max(sizes.buildScratchBufferSize, sizes.refitScratchBufferSize);
        blas->acceleration = [S().device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
        Require(blas->acceleration != nil, "Cannot allocate BLAS");
        AddResource(blas->acceleration, false);
        return BLAS(blas);
    }
}
TLAS CreateTLAS(uint32_t maxCount, bool allowUpdate) {
    @autoreleasepool {
        Require(S().device.supportsRaytracing && maxCount > 0, "Invalid TLAS capacity or unsupported device");
        auto* tlas = new MetalTLAS;
        tlas->allowUpdate = allowUpdate;
        tlas->references.resize(maxCount);
        tlas->instances = CreateBuffer({.name = "TLAS instances", .size = uint64_t(maxCount) * sizeof(MTLIndirectAccelerationStructureInstanceDescriptor),
            .usage = BufferUsage::TransferDst | BufferUsage::AccelerationStructureInput, .memoryType = MemoryType::GPU});
        tlas->count = CreateBuffer({.name = "TLAS instance count", .size = 4, .usage = BufferUsage::TransferDst, .memoryType = MemoryType::GPU});
        tlas->desc = [MTLIndirectInstanceAccelerationStructureDescriptor new];
        tlas->desc.instanceDescriptorBuffer = B(tlas->instances).buffer;
        tlas->desc.instanceCountBuffer = B(tlas->count).buffer;
        tlas->desc.maxInstanceCount = maxCount;
        tlas->desc.instanceDescriptorStride = sizeof(MTLIndirectAccelerationStructureInstanceDescriptor);
        tlas->desc.usage = MTLAccelerationStructureUsagePreferFastBuild;
        if (allowUpdate) tlas->desc.usage |= MTLAccelerationStructureUsageRefit;
        auto sizes = [S().device accelerationStructureSizesWithDescriptor:tlas->desc];
        tlas->scratchBytes = std::max(sizes.buildScratchBufferSize, sizes.refitScratchBufferSize);
        tlas->acceleration = [S().device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
        Require(tlas->acceleration != nil, "Cannot allocate TLAS");
        AddResource(tlas->acceleration, false);
        tlas->resourceid = S().tlas.Allocate();
        [S().arguments setAccelerationStructure:tlas->acceleration atIndex:metal::TLAS_ID + tlas->resourceid];
        return TLAS(tlas);
    }
}
void Cmd::buildBLAS(const std::vector<BLAS>& blases, bool update) {
    @autoreleasepool {
        auto& cmd = C(*this);
        for (const auto& reference : blases) {
            if (!reference) continue;
            auto& blas = A(reference);
            Require(!update || blas.built, "Cannot refit an unbuilt BLAS");
            if (blas.geometry.geometry == GeometryType::Triangles) {
                Compute(cmd);
                uint32_t params[] = {blas.geometry.stride, blas.geometry.triangleCount};
                [cmd.compute setComputePipelineState:S().triangleData];
                [cmd.compute setBuffer:B(blas.geometry.vertices).buffer offset:0 atIndex:0];
                [cmd.compute setBuffer:B(blas.geometry.indices).buffer offset:0 atIndex:1];
                [cmd.compute setBuffer:B(blas.triangleData).buffer offset:0 atIndex:2];
                [cmd.compute setBytes:params length:sizeof(params) atIndex:3];
                [cmd.compute dispatchThreadgroups:MTLSizeMake((params[1] + 63) / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            }
            BuildAcceleration(cmd, blas.acceleration, blas.desc, blas.scratchBytes, update);
            blas.built = true;
        }
    }
}
void Cmd::buildTLAS(const TLAS& reference, const std::vector<BLASInstance>& instances, bool update) {
    @autoreleasepool {
        auto& tlas = T(reference);
        auto& cmd = C(*this);
        Require(instances.size() <= tlas.references.size(), "TLAS instance capacity exceeded");
        Require(!update || (tlas.allowUpdate && tlas.built && instances.size() == tlas.instanceCount), "Invalid TLAS refit");
        uint64_t offset = (cmd.stagingOffset + 255) & ~uint64_t(255);
        uint64_t bytes = instances.size() * sizeof(MTLIndirectAccelerationStructureInstanceDescriptor);
        Require(offset <= STAGING_BYTES && bytes <= STAGING_BYTES - offset, "TLAS upload exceeds the command staging budget");
        auto* data = reinterpret_cast<MTLIndirectAccelerationStructureInstanceDescriptor*>(static_cast<uint8_t*>(cmd.staging.GetPtr()) + offset);
        for (uint32_t index = 0; index < instances.size(); ++index) {
            const auto& instance = instances[index];
            auto& blas = A(instance.blas);
            Require(blas.built, "TLAS references an unbuilt BLAS");
            auto& desc = data[index];
            desc = {};
            for (uint32_t column = 0; column < 4; ++column) {
                for (uint32_t row = 0; row < 3; ++row) desc.transformationMatrix.columns[column].elements[row] = instance.transform[row * 4 + column];
            }
            desc.options = MTLAccelerationStructureInstanceOptionDisableTriangleCulling;
            desc.mask = instance.mask;
            desc.userID = instance.customId;
            desc.accelerationStructureID = blas.acceleration.gpuResourceID;
            tlas.references[index] = instance.blas;
        }
        for (uint32_t index = uint32_t(instances.size()); index < tlas.instanceCount; ++index) tlas.references[index].release();
        cmd.stagingOffset = offset + bytes;
        if (bytes) copy(cmd.staging, tlas.instances, bytes, offset);
        uint32_t count = uint32_t(instances.size());
        copy(&count, tlas.count, sizeof(count));
        BuildAcceleration(cmd, tlas.acceleration, tlas.desc, tlas.scratchBytes, update);
        tlas.instanceCount = count; tlas.built = true;
    }
}
void Shutdown() {
    if (!G) return;
    @autoreleasepool {
        for (auto& cmd : S().commands) if (cmd.pending) CmdWait(cmd.submission);
        S().shuttingDown = true;
        for (auto& cmd : S().commands) {
            cmd.pipeline.release(); cmd.vertices.release(); cmd.indices.release(); cmd.drawableImage.release(); cmd.staging.release();
            cmd.buffer = nil;
        }
        Retire();
        if (S().view.layer == S().layer) S().view.layer = nil;
        State* state = G; delete state; G = nullptr;
    }
}
}
