#pragma once

#include "evk.h"

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#if defined(_DEBUG) || defined(EVK_DEBUG)
#define EVK_ASSERT(cond, message, ...)                                                        \
    if (!(cond)) {                                                                            \
        printf("\033[1;33m %s() \033[1;31m" message "\033[0m", __FUNCTION__, ##__VA_ARGS__);  \
        exit(1);                                                                              \
    }

#define CHECK_VK(cmd) EVK_ASSERT(cmd == VK_SUCCESS, #cmd)  // printf("%s\n", #cmd);
#else
#define EVK_ASSERT(cond, message, ...)
#define CHECK_VK(cmd) cmd
#endif

namespace evk {

    const uint32_t BINDING_STORAGE = 0;
    const uint32_t BINDING_SAMPLER = 1;
    const uint32_t BINDING_IMAGE = 2;
    const uint32_t BINDING_TLAS = 3;

    const uint32_t STORAGE_COUNT = 16384;
    const uint32_t IMAGE_COUNT = 16384;
    const uint32_t SAMPLER_COUNT = IMAGE_COUNT;
    const uint32_t TLAS_COUNT = 16384;
    
    const uint32_t PERF_QUERY_COUNT = 64;
    const uint32_t MAX_COMMAND_BUFFERS = 4;

    enum class CmdState : uint8_t {
        Ready = 0,
        InUse = 1,
        Submitted = 2
    };

    // Forward declarations
    struct CommandBufferData;

    struct CommandBufferData {
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        uint64_t submissionIndex = 0;
        CmdState state = CmdState::Ready;
        bool insideRenderPass = false;
        bool doingPresent = false;

        // Semaphores for swapchain synchronization (when doingPresent is true)
        VkSemaphore imageReadySemaphore = VK_NULL_HANDLE;  // Wait on this before rendering to swapchain
        VkSemaphore cmdDoneSemaphore = VK_NULL_HANDLE;     // Signal this when done, present waits on it

        // Staging buffer for this command buffer
        Buffer stagingBuffer = {};
        uint64_t stagingOffset = 0;

        // Timestamp queries
        VkQueryPool queryPool = VK_NULL_HANDLE;
        std::vector<const char*> timestampNames;
        std::vector<uint64_t> queries;
        std::vector<TimestampEntry> timestampEntries;

        inline int AllocTimestamp(const char* name) {
            int id = int(timestampNames.size());
            timestampNames.push_back(name);
            return id;
        }
    };

    struct SlotAllocator {
        int maxSlot;
        int currentSlot = 0;
        std::vector<int> freeSlots;
        SlotAllocator(int maxSlot) : maxSlot(maxSlot) {
        }

        int alloc() {
            int slot = -1;
            if (freeSlots.size() > 0) {
                slot = freeSlots.back();
                freeSlots.pop_back();
            } else {
                slot = currentSlot++;
                EVK_ASSERT(slot < maxSlot, "Maximum number of slots reached!");
            }
            return slot;
        }
        void free(int slot) {
            freeSlots.push_back(slot);
        }
    };

    struct State {
        VkInstance instance;
        VkPhysicalDevice physicalDevice;
        VkDevice device;
        VkQueue queue;
        uint32_t queueFamily;
        VmaAllocator allocator;
        float timestampPeriod;

        // Descriptors
        VkDescriptorPool descriptorPool;
        VkDescriptorSetLayout descriptorSetLayout;
        VkDescriptorSet descriptorSet;
        VkPipelineLayout pipelineLayout;
        SlotAllocator bufferSlots = SlotAllocator(STORAGE_COUNT);
        SlotAllocator imageSlots = SlotAllocator(IMAGE_COUNT);
        SlotAllocator tlasSlots = SlotAllocator(TLAS_COUNT);

        // Command buffer management (new API)
        std::vector<CommandBufferData> commandBuffers;
        uint64_t nextSubmissionIndex = 1;  // Start at 1 so 0 is invalid
        Cmd currentCmd;  // Current command buffer being recorded
        CommandBufferData* currentCmdData = nullptr;
        
        // Resources pending deletion (waiting for any command buffer to complete)
        std::vector<std::pair<uint64_t, Resource*>> pendingDeletions;

        // Last completed timestamps
        std::vector<TimestampEntry> lastTimestamps;

        // Swapchain
        VkSurfaceKHR surface;
        VkSwapchainKHR swapchain;
        uint32_t swapchainIndex = 0;
        std::vector<Image> swapchainImages;  // Swapchain images

        // Raytracing
        PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR;
        PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR;
        PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR;
        PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR;
        PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR;
        PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR;
        PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR;
        PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR;
        PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR;

        // Pfns
        PFN_vkCmdBeginDebugUtilsLabelEXT vkCmdBeginDebugUtilsLabelEXT;
        PFN_vkCmdEndDebugUtilsLabelEXT vkCmdEndDebugUtilsLabelEXT;
        PFN_vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectNameEXT;
    };
    State& GetState();
    void SetState(State* state);

#define DEFINE_TO_INTERNAL(libClass) \
    static inline Internal_##libClass& ToInternal(const libClass& ref) { return *((Internal_##libClass*)ref.res); }

    struct Internal_Buffer : Resource {
        BufferDesc desc;

        VmaAllocation allocation = {};
        VkBuffer buffer = {};
        VkDeviceAddress deviceAddress = {};
        void* mappedData = {};
        ~Internal_Buffer() {
            vmaDestroyBuffer(GetState().allocator, buffer, allocation);
            // Free descriptor index
            EVK_ASSERT(resourceid != -1, "destroying buffer '%s' with RID = -1", desc.name.c_str());
            GetState().bufferSlots.free(resourceid);
        }
    };
    struct Internal_Image : Resource {
        ImageDesc desc;

        VmaAllocation allocation = {};
        VkImage image = {};
        VkImageView view = {};
        VkSampler sampler = {};

        ~Internal_Image() {
            auto& S = GetState();
            vkDestroySampler(S.device, sampler, nullptr);
            vkDestroyImageView(S.device, view, nullptr);
            if (allocation != nullptr) {
                vmaFreeMemory(S.allocator, allocation);
                vkDestroyImage(S.device, image, nullptr);
                // Free descriptor index
                {
                    EVK_ASSERT(resourceid != -1, "destroying image '%s' with RID = -1", desc.name.c_str());
                    GetState().imageSlots.free(resourceid);
                    resourceid = -1;
                }
            }
        }
    };
    struct Internal_Pipeline : Resource {
        PipelineDesc desc;
        VkPipeline pipeline = {};
        bool isCompute = false;
        ~Internal_Pipeline() {
            vkDestroyPipeline(GetState().device, pipeline, nullptr);
        }
    };
    
    struct Internal_BLAS : Resource {
        std::vector<VkAccelerationStructureGeometryKHR> geometries;
        std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
        std::vector<uint32_t> primCounts;
        VkBuildAccelerationStructureFlagsKHR flags = {};

        VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
        VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};

        VkAccelerationStructureKHR accel = {};
        VkDeviceAddress accStructureDeviceAddress = {};
        Buffer buffer;
        Buffer aabbsBuffer;
        Buffer vertexBuffer;
        Buffer indexBuffer;

        ~Internal_BLAS() {
            GetState().vkDestroyAccelerationStructureKHR(GetState().device, accel, nullptr);
        }
    };
    struct Internal_TLAS : Resource {
        std::vector<VkAccelerationStructureInstanceKHR> instances = {};
        VkAccelerationStructureGeometryKHR geometry = {};

        VkAccelerationStructureBuildGeometryInfoKHR buildInfo;
        VkAccelerationStructureBuildSizesInfoKHR sizeInfo;

        VkAccelerationStructureKHR accel = {};
        Buffer instancesBuffer;
        Buffer buffer;

        ~Internal_TLAS() {
            GetState().vkDestroyAccelerationStructureKHR(GetState().device, accel, nullptr);
            GetState().tlasSlots.free(resourceid);
        }
    };

    DEFINE_TO_INTERNAL(BLAS)
    DEFINE_TO_INTERNAL(TLAS)
    DEFINE_TO_INTERNAL(Image)
    DEFINE_TO_INTERNAL(Buffer)
    DEFINE_TO_INTERNAL(Pipeline)
}  // namespace evk