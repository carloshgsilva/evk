#include <assert.h>

#include "win_dbg.h"
#include "evk.h"

void main_llm();

int main() {
    set_unhandled_exception_filter();

    evk::InitializeEVK({
        .applicationName = "evk_example",
        .applicationVersion = 1,
        .engineName = "evk_example_engine",
        .engineVersion = 1,
        .enableSwapchain = false,
    });

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
    
    evk::Shutdown();

    printf("All tests passed successfully!\n");
    return 0;
}
