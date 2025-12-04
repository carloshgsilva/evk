#include <assert.h>

#include "win_dbg.h"
#include "evk.h"

void main_llm();

int main() {
    printf("[test] Starting tests...\n");
    set_unhandled_exception_filter();

    evk::InitializeEVK({
        .applicationName = "evk_example",
        .applicationVersion = 1,
        .engineName = "evk_example_engine",
        .engineVersion = 1,
        .enableSwapchain = false,
    });

    main_llm();

    // Test the command buffer API
    {
        // Create a test buffer
        evk::Buffer testBuffer = evk::CreateBuffer({
            .name = "Test Buffer",
            .size = 1024,
            .usage = evk::BufferUsage::Storage,
            .memoryType = evk::MemoryType::GPU,
        });

        // Begin a command buffer
        auto& cmd = evk::CmdBegin(evk::Queue::Graphics);
        cmd.fill(testBuffer, 0x12345678, 1024);
        uint64_t idx = cmd.submit();
        
        // Should not be done immediately
        assert(evk::CmdDone(idx) == false);
        
        // Wait for completion
        evk::CmdWait(idx);

        // Now it should be done
        assert(evk::CmdDone(idx) == true);
    }
    
    // Test timestamps
    {
        // Timestamps should be empty initially (or from previous cmd)
        auto& initialTimestamps = evk::CmdTimestamps();
        
        // Create a buffer for testing
        evk::Buffer testBuffer = evk::CreateBuffer({
            .name = "Timestamp Test Buffer",
            .size = 1024 * 1024,  // 1MB for more measurable time
            .usage = evk::BufferUsage::Storage,
            .memoryType = evk::MemoryType::GPU,
        });
        
        // Begin command buffer with timestamps
        auto& cmd = evk::CmdBegin(evk::Queue::Graphics);
        cmd.timestamp("Fill Buffer", [&]() {
            cmd.fill(testBuffer, 0xFFFFFFFF, 1024 * 1024);
        });
        cmd.timestamp("Fill Buffer 2", [&]() {
            cmd.fill(testBuffer, 0xDEADBEEF, 1024 * 1024);
        });
        uint64_t idx = cmd.submit();
        
        // Before completion, timestamps should still be from previous command (empty)
        assert(evk::CmdDone(idx) == false);
        assert(evk::CmdTimestamps().size() == initialTimestamps.size());
        
        // Wait for completion
        evk::CmdWait(idx);
        assert(evk::CmdDone(idx) == true);
        
        // Now timestamps should be available
        auto& timestamps = evk::CmdTimestamps();
        assert(timestamps.size() == 2);

        // Test that CmdDone also triggers timestamp reading when command completes
        auto& cmd2 = evk::CmdBegin(evk::Queue::Graphics);
        cmd2.timestamp("Third Fill", [&]() {
            cmd2.fill(testBuffer, 0xAAAAAAAA, 1024 * 1024);
        });
        uint64_t idx2 = cmd2.submit();
        
        // Busy wait using CmdDone
        while (!evk::CmdDone(idx2)) {
            // Spin
        }
        
        // After CmdDone returns true, timestamps should be updated
        assert(evk::CmdTimestamps().size() == 1);
    }
    
    // Test TLAS / BLAS
    {
        // Simple triangle geometry
        struct Vertex { float x, y, z; };
        Vertex vertices[3] = {
            {0.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
        };
        uint32_t indices[3] = {0, 1, 2};

        // Create staging buffers to upload vertex/index data
        evk::Buffer vStaging = evk::CreateBuffer({
            .name = "Vertex Staging",
            .size = sizeof(vertices),
            .usage = evk::BufferUsage::TransferSrc,
            .memoryType = evk::MemoryType::CPU_TO_GPU,
        });
        evk::Buffer iStaging = evk::CreateBuffer({
            .name = "Index Staging",
            .size = sizeof(indices),
            .usage = evk::BufferUsage::TransferSrc,
            .memoryType = evk::MemoryType::CPU_TO_GPU,
        });
        evk::WriteBuffer(vStaging, vertices, sizeof(vertices));
        evk::WriteBuffer(iStaging, indices, sizeof(indices));

        // Create GPU-local buffers for acceleration structure input
        evk::Buffer vBuffer = evk::CreateBuffer({
            .name = "Vertex Buffer",
            .size = sizeof(vertices),
            .usage = evk::BufferUsage::Vertex | evk::BufferUsage::AccelerationStructureInput,
            .memoryType = evk::MemoryType::GPU,
        });
        evk::Buffer iBuffer = evk::CreateBuffer({
            .name = "Index Buffer",
            .size = sizeof(indices),
            .usage = evk::BufferUsage::Index | evk::BufferUsage::AccelerationStructureInput,
            .memoryType = evk::MemoryType::GPU,
        });
        assert(vBuffer);
        assert(iBuffer);
        assert(vBuffer.GetReference() != 0);
        assert(iBuffer.GetReference() != 0);

        // Copy staging -> GPU buffers
        auto& copyCmd = evk::CmdBegin(evk::Queue::Graphics);
        copyCmd.copy(vStaging, vBuffer, sizeof(vertices));
        copyCmd.copy(iStaging, iBuffer, sizeof(indices));
        uint64_t copyIdx = copyCmd.submit();
        evk::CmdWait(copyIdx);

        // Create BLAS
        evk::BLAS blas = evk::CreateBLAS({
            .geometry = evk::GeometryType::Triangles,
            .stride = sizeof(Vertex),
            .vertices = vBuffer,
            .vertexCount = 3,
            .indices = iBuffer,
            .triangleCount = 1,
        });
        assert(blas);

        // Create TLAS with some slack in max instances so the library check passes
        evk::TLAS tlas = evk::CreateTLAS(4, true);
        assert(tlas);

        // Build BLAS then TLAS
        auto& buildCmd = evk::CmdBegin(evk::Queue::Graphics);
        buildCmd.buildBLAS({blas});
        std::vector<evk::BLASInstance> instances(1);
        instances[0].blas = blas;
        instances[0].customId = 42;
        buildCmd.buildTLAS(tlas, instances);
        uint64_t buildIdx = buildCmd.submit();
        evk::CmdWait(buildIdx);

        // Now update TLAS transform and rebuild (update=true)
        for (int i =0 ; i < 16; i++) {
            auto& updateCmd = evk::CmdBegin(evk::Queue::Graphics);
            updateCmd.buildTLAS(tlas, instances, true);
            uint64_t updateIdx = updateCmd.submit();
            evk::CmdWait(updateIdx);
        }
    }

    evk::Shutdown();
    printf("[test] All tests passed successfully!\n");
    return 0;
}
