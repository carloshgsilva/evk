# EVK - Easy Vulkan

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modern, easy-to-use C++ wrapper for Vulkan 1.3 that simplifies graphics programming while maintaining high performance and full access to Vulkan's capabilities.

## Features

### 🚀 Core Features
- **Vulkan 1.3 Support** - Full support for the latest Vulkan API features
- **Modern C++20** - Leverages modern C++ features for cleaner, safer code
- **Automatic Resource Management** - Smart pointers and RAII for Vulkan resources
- **Memory Management** - Integrated Vulkan Memory Allocator (VMA) for optimal GPU memory usage
- **Multi-threaded** - Designed for modern multi-threaded applications

### 🎨 Graphics Pipeline
- **Graphics & Compute Pipelines** - Support for both graphics and compute workloads
- **Shader Support** - Direct shader bytecode loading for vertex, fragment, and compute shaders
- **Render Passes** - Simplified render pass management with automatic layout transitions
- **Depth/Stencil Testing** - Full depth and stencil buffer support
- **Blending Modes** - Alpha blending and additive blending support
- **Face Culling** - Front, back, and no culling options

### 📦 Resource Management
- **Buffers** - Vertex, index, uniform, storage, and indirect buffers
- **Images/Textures** - 2D/3D textures with mipmapping and array layers
- **Memory Types** - CPU, GPU, CPU-to-GPU, and GPU-to-CPU memory configurations
- **Resource Barriers** - Automatic and manual pipeline barriers for synchronization

### ⚡ Advanced Features
- **Ray Tracing** - Built-in support for Vulkan ray tracing with BLAS/TLAS
- **ImGui Integration** - Ready-to-use ImGui backend for Vulkan
- **Timestamp Queries** - GPU timing and profiling support
- **Memory Budget Tracking** - Monitor GPU memory usage and budgets
- **Indirect Drawing** - Support for indirect draw calls and compute dispatches

### 🧮 Math Library
- **GLSL-compatible Types** - vec2, vec3, vec4, mat4, quat with operator overloading
- **Matrix Operations** - Compose, decompose, inverse, transpose, look-at
- **Quaternion Support** - Rotation, slerp, and matrix conversion
- **Utility Functions** - dot, cross, normalize, distance, and more

## Quick Start

### Basic Setup

```cpp
#include "evk.h"

int main() {
    // Initialize EVK
    evk::EvkDesc desc = {
        .applicationName = "My Vulkan App",
        .applicationVersion = 1,
        .engineName = "EVK",
        .engineVersion = 1,
        .enableSwapchain = true  // Enable for windowed applications
    };

    if (!evk::InitializeEVK(desc)) {
        return -1;
    }

    // Your rendering code here
    while (running) {
        // Begin frame
        evk::CmdBeginPresent();

        // Render commands
        // ...

        // End frame
        evk::CmdEndPresent();
        evk::Submit();
    }

    evk::Shutdown();
    return 0;
}
```

### Creating Resources

```cpp
// Create a vertex buffer
evk::BufferDesc bufferDesc = {
    .name = "VertexBuffer",
    .size = sizeof(vertices),
    .usage = evk::BufferUsage::Vertex,
    .memoryType = evk::MemoryType::GPU
};
evk::Buffer vertexBuffer = evk::CreateBuffer(bufferDesc);

// Upload data to buffer
evk::WriteBuffer(vertexBuffer, vertices.data(), sizeof(vertices));

// Create an image/texture
evk::ImageDesc imageDesc = {
    .name = "Texture",
    .extent = {512, 512},
    .format = evk::Format::RGBA8Unorm,
    .usage = evk::ImageUsage::Sampled,
    .filter = evk::Filter::Linear
};
evk::Image texture = evk::CreateImage(imageDesc);
```

### Graphics Pipeline

```cpp
evk::PipelineDesc pipelineDesc = {
    .name = "BasicPipeline",
    .VS = vertexShaderBytecode,
    .FS = fragmentShaderBytecode,
    .bindings = {{evk::Format::RGBA32Sfloat}},  // Vertex attributes
    .attachments = {evk::Format::RGBA8Unorm},   // Color attachments
    .blends = {evk::Blend::Alpha},             // Blending modes
    .primitive = evk::Primitive::Triangle,
    .cull = evk::Cull::Back
};

evk::Pipeline pipeline = evk::CreatePipeline(pipelineDesc);

// Bind and use the pipeline
evk::CmdBind(pipeline);
```

### Rendering

```cpp
// Render to a framebuffer
evk::Image colorAttachment = /* your render target */;
evk::Image depthAttachment = /* optional depth buffer */;

evk::CmdRender({colorAttachment}, {clearColor}, [&]() {
    evk::CmdBind(pipeline);
    evk::CmdVertex(vertexBuffer);
    evk::CmdIndex(indexBuffer);
    evk::CmdDrawIndexed(indexCount);
});
```

### Compute Pipeline

```cpp
evk::PipelineDesc computeDesc = {
    .name = "ComputePipeline",
    .CS = computeShaderBytecode
};

evk::Pipeline computePipeline = evk::CreatePipeline(computeDesc);

evk::CmdBind(computePipeline);
evk::CmdDispatch(groupCountX, groupCountY, groupCountZ);
```

## Building

### Prerequisites
- CMake 3.8 or higher
- Vulkan SDK
- C++20 compatible compiler

### Build Steps

```bash
# Clone the repository
git clone https://github.com/carloshgsilva/evk
cd evk

# Configure and build
mkdir build && cd build
cmake ..
cmake --build . --config Release

# The library will be available as evk.lib (Windows) or libevk.a (Linux/macOS)
```

### Integration

Add the following to your CMakeLists.txt:

```cmake
# Add EVK as a subdirectory
add_subdirectory(path/to/evk)

# Link against EVK
target_link_libraries(your_target evk)

# Include EVK headers
target_include_directories(your_target PRIVATE path/to/evk)
```

## Architecture

EVK is designed with the following principles:

### 🏗️ Clean Architecture
- **Single Header Interface** - Everything accessible through `evk.h`
- **Resource IDs** - Efficient resource management with integer IDs
- **RAII Pattern** - Automatic resource cleanup with smart pointers
- **Command Buffer Recording** - Vulkan-style command recording for efficiency

### 🔧 Implementation Details
- **Vulkan Memory Allocator** - Integrated VMA for optimal memory management
- **Format Conversion** - Automatic conversion between EVK and Vulkan formats
- **Pipeline State Management** - Cached pipeline states for performance
- **Synchronization** - Automatic barrier insertion and queue synchronization

### 📊 Performance Features
- **Memory Pooling** - Efficient memory allocation and reuse
- **Descriptor Set Management** - Optimized descriptor set updates
- **Command Buffer Reuse** - Minimizes allocation overhead
- **GPU Timestamp Queries** - Built-in profiling and performance monitoring

## Advanced Usage

### Ray Tracing

```cpp
// Create Bottom-Level Acceleration Structure
evk::BLASDesc blasDesc = {
    .geometry = evk::GeometryType::Triangles,
    .vertices = vertexBuffer,
    .indices = indexBuffer,
    .triangleCount = triangleCount
};
evk::BLAS blas = evk::CreateBLAS(blasDesc);

// Create Top-Level Acceleration Structure
evk::TLAS tlas = evk::CreateTLAS(maxInstanceCount, allowUpdates);

// Build acceleration structures
std::vector<evk::BLASInstance> instances = {/* instance data */};
evk::CmdBuildBLAS({blas});
evk::CmdBuildTLAS(tlas, instances);
```

### ImGui Integration

```cpp
#include "imgui_impl_evk.h"

// Initialize ImGui
ImGui::CreateContext();
ImGui_ImplEvk_Init();

// In your render loop
ImGui_ImplEvk_PrepareRender(ImGui::GetDrawData());
ImGui_ImplEvk_RenderDrawData(ImGui::GetDrawData());
```

### Memory Management

```cpp
// Get current memory budget
evk::MemoryBudget budget = evk::GetMemoryBudget();

// Monitor memory usage
for (int i = 0; i < budget.MAX_HEAPS; ++i) {
    printf("Heap %d: %llu/%llu bytes used\n",
           i, budget.heaps[i].usage, budget.heaps[i].budget);
}
```

### Profiling

```cpp
// Profile a section of GPU work
evk::CmdTimestamp("ComputePass", [&]() {
    evk::CmdBind(computePipeline);
    evk::CmdDispatch(workGroups);
});

// Get timing results
auto timestamps = evk::GetTimestamps();
for (const auto& ts : timestamps) {
    printf("%s: %.3f ms\n", ts.name, ts.end - ts.start);
}
```

## Examples

See the `examples/` directory for complete working applications demonstrating:
- Basic triangle rendering
- Texture mapping
- Compute shader usage
- Ray tracing
- ImGui integration
- Performance monitoring

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

### Development Setup

```bash
# Enable debug mode for development
cmake -DEVK_DEBUG=ON ..
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of the excellent [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
- Inspired by modern graphics APIs and game engines
- Thanks to the Vulkan community for documentation and examples

## Roadmap

- [ ] Additional shader stages support
- [ ] Enhanced debugging features
- [ ] More examples and tutorials
- [ ] Performance optimizations
- [ ] Additional platform support
- [ ] Extended ray tracing features

---

**Note**: This library is currently in active development. APIs may change between versions.