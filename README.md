# EVK

A lightweight C++20 wrapper for Vulkan 1.3 focused on simplicity and performance.

## Overview

EVK provides a clean abstraction over the Vulkan API while maintaining direct access to GPU capabilities. It handles the boilerplate of Vulkan setup, resource management, and synchronization, letting you focus on your application logic.

**Key characteristics:**
- Single header interface (`evk.h`)
- RAII-based resource management
- Integrated memory allocation via VMA
- Support for graphics, compute, and ray tracing pipelines

## Requirements

- C++20 compiler
- Vulkan SDK
- CMake 3.8+

## Building

```bash
git clone https://github.com/carloshgsilva/evk
cd evk
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

To integrate into your project:

```cmake
add_subdirectory(path/to/evk)
target_link_libraries(your_target evk)
```

## Usage

### Initialization

```cpp
#include "evk.h"

int main() {
    evk::EvkDesc desc = {
        .applicationName = "MyApp",
        .applicationVersion = 1,
        .engineName = "EVK",
        .engineVersion = 1,
        .enableSwapchain = true
    };

    if (!evk::InitializeEVK(desc)) {
        return -1;
    }

    // Application loop
    while (running) {
        auto& cmd = evk::CmdBegin(evk::Queue::Graphics);
        cmd.present([&]() {
            // ... render commands using cmd.bind(), cmd.draw(), etc ...
        });
        uint64_t idx = cmd.submit();
        // evk::CmdWait(idx); // wait for GPU to finish (optional)
    }

    evk::Shutdown();
    return 0;
}
```

### Buffers and Images

```cpp
// Create a GPU buffer
evk::Buffer buffer = evk::CreateBuffer({
    .name = "VertexBuffer",
    .size = dataSize,
    .usage = evk::BufferUsage::Vertex,
    .memoryType = evk::MemoryType::GPU
});

evk::WriteBuffer(buffer, data, dataSize);

// Create an image
evk::Image texture = evk::CreateImage({
    .name = "Texture",
    .extent = {512, 512},
    .format = evk::Format::RGBA8Unorm,
    .usage = evk::ImageUsage::Sampled,
    .filter = evk::Filter::Linear
});
```

### Graphics Pipeline

```cpp
evk::Pipeline pipeline = evk::CreatePipeline({
    .name = "MainPipeline",
    .VS = evk::loadSpirvFile("shaders/vert.spv"),
    .FS = evk::loadSpirvFile("shaders/frag.spv"),
    .bindings = {{evk::Format::RGB32Sfloat, evk::Format::RG32Sfloat}},
    .attachments = {evk::Format::RGBA8Unorm},
    .blends = {evk::Blend::Alpha},
    .primitive = evk::Primitive::Triangle,
    .cull = evk::Cull::Back,
    .depthTest = true,
    .depthWrite = true,
    .depthOp = evk::Op::Less
});
```

### Rendering

The `CmdRender` function handles image layout transitions automatically:

```cpp
auto& cmd = evk::CmdBegin(evk::Queue::Graphics);
cmd.render({colorTarget}, {ClearColor{{0.0f, 0.0f, 0.0f, 1.0f}}}, [&]() {
    cmd.bind(pipeline);
    cmd.vertex(vertexBuffer);
    cmd.index(indexBuffer);
    cmd.viewport(0, 0, width, height);
    cmd.scissor(0, 0, width, height);
    cmd.push(pushConstants);
    cmd.drawIndexed(indexCount);
});
uint64_t idx = cmd.submit();
evk::CmdWait(idx);
```

### Compute Pipeline

```cpp
evk::Pipeline compute = evk::CreatePipeline({
    .name = "ComputeShader",
    .CS = evk::loadSpirvFile("shaders/compute.spv")
});

auto& cmd = evk::CmdBegin(evk::Queue::Graphics);
cmd.bind(compute);
cmd.dispatch(groupsX, groupsY, groupsZ);
cmd.barrier();  // Synchronize compute output
uint64_t idx = cmd.submit();
evk::CmdWait(idx);
```

### Push Constants

EVK provides a type-safe way to pass push constants:

```cpp
struct PushData {
    glsl::mat4 viewProj;
    glsl::vec4 lightPos;
};

PushData push = { /* ... */ };
{
    auto& cmd = evk::CmdBegin(evk::Queue::Graphics);
    cmd.push(push);
    uint64_t idx = cmd.submit();
    evk::CmdWait(idx);
}

// Or using the Constant helper for mixed types:
{
    auto& cmd2 = evk::CmdBegin(evk::Queue::Graphics);
    cmd2.push(evk::Constant{
        buffer.GetReference(),
        textureIndex,
        time
    });
    uint64_t idx2 = cmd2.submit();
    evk::CmdWait(idx2);
}
```

### Image Operations

```cpp
auto& cmd = evk::CmdBegin(evk::Queue::Graphics);
// Layout transitions
cmd.barrier(image, evk::ImageLayout::Undefined, evk::ImageLayout::TransferDst);

// Copy between images
cmd.copy(srcImage, dstImage);

// Copy buffer to image
cmd.copy(stagingBuffer, texture);

// Blit with scaling
cmd.blit(src, dst, srcRegion, dstRegion, evk::Filter::Linear);

uint64_t idx = cmd.submit();
evk::CmdWait(idx);
```

### Ray Tracing

```cpp
// Create acceleration structures
evk::BLAS blas = evk::CreateBLAS({
    .geometry = evk::GeometryType::Triangles,
    .stride = sizeof(Vertex),
    .vertices = vertexBuffer,
    .vertexCount = vertexCount,
    .indices = indexBuffer,
    .triangleCount = triangleCount
});

evk::TLAS tlas = evk::CreateTLAS(maxInstances, allowUpdate);

// Build
auto& cmd = evk::CmdBegin(evk::Queue::Graphics);
cmd.buildBLAS({blas});
cmd.buildTLAS(tlas, instances);
uint64_t idx = cmd.submit();
evk::CmdWait(idx);
```

### GPU Profiling

```cpp
auto& cmd = evk::CmdBegin(evk::Queue::Graphics);
cmd.timestamp("RenderPass", [&]() {
    // ... rendering code ...
});
uint64_t idx = cmd.submit();
evk::CmdWait(idx);

for (const auto& ts : evk::GetTimestamps()) {
    printf("%s: %.2f ms\n", ts.name, ts.end - ts.start);
}
```

### Memory Monitoring

```cpp
evk::MemoryBudget budget = evk::GetMemoryBudget();
for (uint32_t i = 0; i < budget.MAX_HEAPS; ++i) {
    if (budget.heaps[i].budget > 0) {
        printf("Heap %u: %llu / %llu MB\n", i,
               budget.heaps[i].usage / (1024*1024),
               budget.heaps[i].budget / (1024*1024));
    }
}
```

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `InitializeEVK(desc)` | Initialize the Vulkan context |
| `Shutdown()` | Clean up all resources |
| `CmdBegin(queue) -> Cmd&` | Begin recording on a queue and return a command buffer handle |
| `CmdDone(submissionIndex)` | Check if a submitted command buffer has finished |
| `CmdWait(submissionIndex)` | Wait for a submitted command buffer to complete |

### Resource Creation

| Function | Description |
|----------|-------------|
| `CreateBuffer(desc)` | Create a buffer (vertex, index, storage, etc.) |
| `CreateImage(desc)` | Create an image/texture |
| `CreatePipeline(desc)` | Create a graphics or compute pipeline |
| `CreateBLAS(desc)` | Create bottom-level acceleration structure |
| `CreateTLAS(count, update)` | Create top-level acceleration structure |

### Command Recording

| Function | Description |
|----------|-------------|
| `Cmd::bind(pipeline)` | Bind a pipeline for subsequent draw/dispatch calls |
| `Cmd::vertex(buffer)` | Bind vertex buffer |
| `Cmd::index(buffer)` | Bind index buffer |
| `Cmd::push(data)` | Set push constant data |
| `Cmd::draw(...)` | Issue draw call |
| `Cmd::drawIndexed(...)` | Issue indexed draw call |
| `Cmd::dispatch(x, y, z)` | Dispatch compute work |
| `Cmd::render(attachments, clears, fn)` | Execute render pass |
| `Cmd::barrier(...)` | Insert pipeline barrier |
| `Cmd::copy(...)` | Copy between buffers/images |
| `Cmd::submit()` | Submit the recorded command buffer to the GPU and return a submission index |
| `Cmd::present(fn)` | Convenience to begin/end rendering to swapchain and execute fn |
| `Cmd::timestamp(name, fn)` | Insert GPU timestamp region around fn and record timing |

### Enums

**Format**: `R8Uint`, `R16Uint`, `R32Uint`, `RGBA8Unorm`, `RGBA16Sfloat`, `RGBA32Sfloat`, `D32Sfloat`, etc.

**BufferUsage**: `TransferSrc`, `TransferDst`, `Vertex`, `Index`, `Indirect`, `Storage`, `AccelerationStructure`

**ImageUsage**: `TransferSrc`, `TransferDst`, `Sampled`, `Attachment`, `Storage`

**MemoryType**: `GPU`, `CPU_TO_GPU`, `GPU_TO_CPU`

**Blend**: `Disabled`, `Alpha`, `Additive`

**Cull**: `None`, `Front`, `Back`

## ImGui Integration

EVK includes a ready-to-use ImGui backend:

```cpp
#include "imgui_impl_evk.h"

ImGui::CreateContext();
ImGui_ImplEvk_Init();

// In render loop:
ImGui_ImplEvk_PrepareRender(ImGui::GetDrawData());
ImGui_ImplEvk_RenderDrawData(ImGui::GetDrawData());
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Built with [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator).