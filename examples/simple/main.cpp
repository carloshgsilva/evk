#include <iostream>
#include <evk.h>

int main() {
    evk::EvkDesc desc;
    desc.applicationName = "evk_example";
    desc.applicationVersion = 1;
    desc.engineName = "evk_example_engine";
    desc.engineVersion = 1;
    desc.enableSwapchain = false; // keep simple: no surface

    if (!evk::InitializeEVK(desc)) {
        std::cerr << "Failed to initialize EVK" << std::endl;
        return -1;
    }

    // Query some properties to ensure linking works
    uint32_t frames = evk::GetFrameBufferingCount();
    std::cout << "EVK initialized with frame buffering count: " << frames << std::endl;

    evk::Shutdown();
    return 0;
}
