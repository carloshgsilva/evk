#include "evk_vulkan.h"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
#include "evk.h"

namespace evk {
    static State* GState;

    const VmaMemoryUsage MEMORY_TYPE_VMA[] = {
        VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_ONLY,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
        VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
        VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_TO_CPU,
    };
    struct FormatConversion {
        const char* name;
        Format evkFormat;
        VkFormat format;
        size_t size;
    };
#define FMT_C(name, vkFormat, size)           \
    {                                         \
#name, Format::name, vkFormat, size \
    }
    const FormatConversion FORMAT_VK[] = {
        FMT_C(Undefined, VK_FORMAT_UNDEFINED, 1),
        FMT_C(R8Uint, VK_FORMAT_R8_UINT, 1),
        FMT_C(R16Uint, VK_FORMAT_R16_UINT, 2),
        FMT_C(R32Uint, VK_FORMAT_R32_UINT, 4),
        FMT_C(R64Uint, VK_FORMAT_R64_UINT, 8),
        FMT_C(BGRA8Unorm, VK_FORMAT_B8G8R8A8_UNORM, 4),
        FMT_C(BGRA8Snorm, VK_FORMAT_B8G8R8A8_SNORM, 4),
        FMT_C(RGBA8Unorm, VK_FORMAT_R8G8B8A8_UNORM, 4),
        FMT_C(RGBA8Snorm, VK_FORMAT_R8G8B8A8_SNORM, 4),
        FMT_C(RG16Sfloat, VK_FORMAT_R16G16_SFLOAT, 4),
        FMT_C(RGBA16Sfloat, VK_FORMAT_R16G16B16A16_SFLOAT, 8),
        FMT_C(RGBA16Unorm, VK_FORMAT_R16G16B16A16_UNORM, 8),
        FMT_C(RGBA16Snorm, VK_FORMAT_R16G16B16A16_SNORM, 8),
        FMT_C(R32Sfloat, VK_FORMAT_R32_SFLOAT, 4),
        FMT_C(RG32Sfloat, VK_FORMAT_R32G32_SFLOAT, 8),
        FMT_C(RGB32Sfloat, VK_FORMAT_R32G32B32_SFLOAT, 12),
        FMT_C(RGBA32Sfloat, VK_FORMAT_R32G32B32A32_SFLOAT, 16),
        FMT_C(RGBA32Sint, VK_FORMAT_R32G32B32A32_SINT, 16),
        FMT_C(RGBA32Uint, VK_FORMAT_R32G32B32A32_UINT, 16),
        FMT_C(D24UnormS8Uint, VK_FORMAT_D24_UNORM_S8_UINT, 4),
        FMT_C(D32Sfloat, VK_FORMAT_D32_SFLOAT, 4),
        // FMT_C(Bc1RgbaUnormBlock,    1)
    };
    int _GetFormatByName(const char* name) {
        constexpr size_t COUNT = sizeof(FORMAT_VK) / sizeof(FormatConversion);
        for (int i = 0; i < COUNT; i++) {
            if (!std::strcmp(name, FORMAT_VK[i].name)) {
                return i;
            }
        }
        return -1;
    }
    const VkFilter FILTER_VK[] = {VK_FILTER_NEAREST, VK_FILTER_LINEAR};
    const VkPipelineStageFlagBits STAGE_VK[] = {
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,              // TopOfPipe,
        VK_PIPELINE_STAGE_HOST_BIT,                     // Host,
        VK_PIPELINE_STAGE_TRANSFER_BIT,                 // Transfer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,           // Compute,
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,            // DrawIndirect,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,             // VertexInput,
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,            // VertexShader,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,     // EarlyFragmentTest,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,          // FragmentShader,
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,      // LateFragmentTest,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,  // ColorAttachmentOutput,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,           // BottomOfPipe,
        VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,             // AllGraphics,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,             // AllCommands
    };
    const VkImageLayout IMAGE_LAYOUT_VK[] = {
        VK_IMAGE_LAYOUT_UNDEFINED,                 // Undefined,
        VK_IMAGE_LAYOUT_GENERAL,                   // General,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,      // TransferSrc,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,      // TransferDst,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,  // ShaderRead,
        VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR,    // Attachment,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,           // Present
    };
    const VkPrimitiveTopology PRIMITIVE_TOPOLOGY_VK[] = {VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_LIST};
    const VkCullModeFlags TO_VK_CULL_MODE[] = {VK_CULL_MODE_NONE, VK_CULL_MODE_FRONT_BIT, VK_CULL_MODE_BACK_BIT};

    bool DoesFormatHaveDepth(Format format) {
        switch (format) {
            case Format::D24UnormS8Uint:
                return true;
            case Format::D32Sfloat:
                return true;
            default:
                return false;
        }
    }

    bool DoesFormatHaveStencil(Format format) {
        switch (format) {
            case Format::D24UnormS8Uint:
                return true;
            default:
                return false;
        }
    }

    bool RecreateSwapchain();

    void Resource::decRef() {
        refCount--;
        if (refCount == 0) {
            GetFrame().toDelete.push_back(this);
        }
    }
    RID ResourceRef::GetRID() const {
        EVK_ASSERT(res != nullptr, "Trying to get resource id of invalid resource");

        EVK_ASSERT(!dynamic_cast<Internal_Image*>(res) || ((uint32_t) dynamic_cast<Internal_Image*>(res)->desc.usage & (uint32_t)ImageUsage::Sampled || (uint32_t) dynamic_cast<Internal_Image*>(res)->desc.usage & (uint32_t)ImageUsage::Storage),
                   "Image '%s' must have ImageUsage::Sampled or ImageUsage::Storage", dynamic_cast<Internal_Image*>(res)->desc.name.c_str());

        EVK_ASSERT(!dynamic_cast<Internal_Buffer*>(res) || ((uint32_t) dynamic_cast<Internal_Buffer*>(res)->desc.usage & (uint32_t)BufferUsage::Storage), "Buffer '%s' must have BufferUsage::Storage",
                   dynamic_cast<Internal_Buffer*>(res)->desc.name.c_str());

        return res->resourceid;
    }

    ////////////
    // Buffer //
    ////////////
    Buffer CreateBuffer(const BufferDesc& desc) {
        auto& S = GetState();
        Internal_Buffer* res = new Internal_Buffer();

        EVK_ASSERT(desc.size != 0, "Buffer size must not be zero");

        // Immutable state
        res->desc = desc;

        VkBufferUsageFlags usageBits{};
        if ((int)(desc.usage | BufferUsage::TransferSrc)) usageBits |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if ((int)(desc.usage | BufferUsage::TransferDst)) usageBits |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if ((int)(desc.usage | BufferUsage::Vertex)) usageBits |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        if ((int)(desc.usage | BufferUsage::Index)) usageBits |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        if ((int)(desc.usage | BufferUsage::Indirect)) usageBits |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        if ((int)(desc.usage | BufferUsage::Storage)) usageBits |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if ((int)(desc.usage | BufferUsage::AccelerationStructure)) usageBits |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
        if ((int)(desc.usage | BufferUsage::AccelerationStructureInput)) usageBits |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        if (desc.memoryType == MemoryType::GPU || desc.memoryType == MemoryType::CPU_TO_GPU) {
            usageBits |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        }

        VkBufferCreateInfo buffci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffci.size = desc.size;
        buffci.queueFamilyIndexCount = 1;
        buffci.pQueueFamilyIndices = &S.queueFamily;
        buffci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        buffci.usage = usageBits;

        VmaAllocationCreateInfo allocCreateInfo = {};
        allocCreateInfo.usage = MEMORY_TYPE_VMA[(int)desc.memoryType];
        allocCreateInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo allocInfo;

        vmaCreateBuffer(S.allocator, &buffci, &allocCreateInfo, (VkBuffer*)&res->buffer, &res->allocation, &allocInfo);
        res->mappedData = allocInfo.pMappedData;

        if (desc.memoryType == MemoryType::GPU || desc.memoryType == MemoryType::CPU_TO_GPU) {
            VkBufferDeviceAddressInfo addressInfo = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                .buffer = res->buffer,
            };
            res->deviceAddress = vkGetBufferDeviceAddress(S.device, &addressInfo);
        }

#if EVK_DEBUG
        VkDebugUtilsObjectNameInfoEXT name = {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .objectType = VkObjectType::VK_OBJECT_TYPE_BUFFER,
            .objectHandle = (uint64_t)(VkBuffer)res->buffer,
            .pObjectName = desc.name.c_str(),
        };
        GetState().vkSetDebugUtilsObjectNameEXT(GetState().device, &name);
#endif

        // Alloc descriptor index
        {
            res->resourceid = S.bufferSlots.alloc();

            VkDescriptorBufferInfo info = {};
            info.buffer = res->buffer;
            info.offset = 0;
            info.range = res->desc.size;

            VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo = &info;
            write.dstSet = S.descriptorSet;
            write.dstBinding = BINDING_STORAGE;
            write.dstArrayElement = res->resourceid;

            vkUpdateDescriptorSets(S.device, 1, &write, 0, nullptr);
        }

        return Buffer(res);
    }
    void* Buffer::GetPtr() {
        Internal_Buffer* buffer = dynamic_cast<Internal_Buffer*>(res);
        EVK_ASSERT(buffer, "Invalid Buffer");
        EVK_ASSERT(buffer->desc.memoryType != MemoryType::GPU, "Trying to write to buffer '%s', but its memory type is GPU!", buffer->desc.name.c_str());
        return buffer->mappedData;
    }
    void WriteBuffer(Buffer& buffer, void* src, uint64_t size, uint64_t offset) {
        Internal_Buffer* res = (Internal_Buffer*)buffer.res;
        EVK_ASSERT(GetDesc(buffer).memoryType != MemoryType::GPU, "Trying to write to buffer '%s', but its memory type is GPU!", GetDesc(buffer).name.c_str());
        EVK_ASSERT(offset + size <= res->desc.size, "Trying to write Buffer '%s' out of range!", res->desc.name.c_str());
        std::memcpy((void*)((size_t)res->mappedData + offset), src, size);
    }
    void ReadBuffer(Buffer& buffer, void* dst, uint64_t size, uint64_t offset) {
        Internal_Buffer* res = (Internal_Buffer*)buffer.res;
        EVK_ASSERT(GetDesc(buffer).memoryType != MemoryType::GPU, "Trying to read from buffer '%s', but its memory type is GPU!", GetDesc(buffer).name.c_str());
        EVK_ASSERT(offset + size <= res->desc.size, "Trying to read Buffer '%s' out of range!", res->desc.name.c_str());
        std::memcpy(dst, (void*)((size_t)res->mappedData + offset), size);
    }

    ///////////
    // Image //
    ///////////
    void InitializeImageView(Internal_Image* state) {
        VkImageAspectFlags aspects = {};
        if (DoesFormatHaveDepth(state->desc.format)) {
            aspects |= VK_IMAGE_ASPECT_DEPTH_BIT;
        } else {
            aspects |= VK_IMAGE_ASPECT_COLOR_BIT;
        }

        // ImageView
        VkImageViewCreateInfo viewci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewci.image = state->image;
        viewci.format = FORMAT_VK[(int)state->desc.format].format;
        viewci.components.r = VK_COMPONENT_SWIZZLE_R;
        viewci.components.g = VK_COMPONENT_SWIZZLE_G;
        viewci.components.b = VK_COMPONENT_SWIZZLE_B;
        viewci.components.a = VK_COMPONENT_SWIZZLE_A;
        viewci.viewType = state->desc.isCube ? VK_IMAGE_VIEW_TYPE_CUBE : (state->desc.extent.depth == 1 ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_3D);
        viewci.subresourceRange.aspectMask = aspects;
        viewci.subresourceRange.baseMipLevel = 0;
        viewci.subresourceRange.levelCount = state->desc.mipCount;
        viewci.subresourceRange.baseArrayLayer = 0;
        viewci.subresourceRange.layerCount = state->desc.layerCount;
        CHECK_VK(vkCreateImageView(GetState().device, &viewci, nullptr, &state->view));

        // Sampler
        if ((int)state->desc.usage & (int)ImageUsage::Sampled) {
            VkSamplerCreateInfo samplerci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
            samplerci.minFilter = FILTER_VK[(int)state->desc.filter];
            samplerci.magFilter = FILTER_VK[(int)state->desc.filter];
            samplerci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerci.anisotropyEnable = false;
            samplerci.maxAnisotropy = 0.0f;
            samplerci.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
            samplerci.unnormalizedCoordinates = false;
            samplerci.compareEnable = false;
            samplerci.compareOp = VK_COMPARE_OP_ALWAYS;
            samplerci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            samplerci.minLod = 0;
            samplerci.maxLod = float(state->desc.mipCount);
            CHECK_VK(vkCreateSampler(GetState().device, &samplerci, nullptr, &state->sampler));
        }
    }
    Image CreateImage(const ImageDesc& desc) {
        auto& S = GetState();
        Internal_Image* res = new Internal_Image();

        EVK_ASSERT(desc.format != Format::Undefined, "Image '%s' format is Undefined, did you forgot to set the format?", desc.name.c_str());

        // Immutable state
        res->desc = desc;

        // Usage flags
        ImageUsage usage = desc.usage;
        VkImageUsageFlags usageBits{};
        usageBits |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        usageBits |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if ((int)usage & (int)ImageUsage::Sampled) usageBits |= VK_IMAGE_USAGE_SAMPLED_BIT;
        if ((int)usage & (int)ImageUsage::Attachment) usageBits |= DoesFormatHaveDepth(desc.format) ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if ((int)usage & (int)ImageUsage::Storage) usageBits |= VK_IMAGE_USAGE_STORAGE_BIT;

        VkImageCreateInfo imageci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imageci.imageType = desc.extent.depth == 1 ? VK_IMAGE_TYPE_2D : VK_IMAGE_TYPE_3D;
        imageci.format = FORMAT_VK[(int)desc.format].format;
        imageci.usage = usageBits;
        imageci.extent = {desc.extent.width, desc.extent.height, desc.extent.depth};
        imageci.mipLevels = desc.mipCount;
        imageci.arrayLayers = desc.layerCount;
        imageci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageci.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageci.samples = VK_SAMPLE_COUNT_1_BIT;
        imageci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageci.flags = desc.isCube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;

        VmaAllocationCreateInfo allocCreateInfo = {};
        allocCreateInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        VmaAllocationInfo allocInfo;

        if (vmaCreateImage(S.allocator, &imageci, &allocCreateInfo, (VkImage*)&res->image, &res->allocation, &allocInfo) != VK_SUCCESS) {
            EVK_ASSERT(false, "Failed to create image '%s'!", desc.name.c_str());
        }

#if EVK_DEBUG
        VkDebugUtilsObjectNameInfoEXT name = {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .objectType = VkObjectType::VK_OBJECT_TYPE_IMAGE,
            .objectHandle = (uint64_t)(VkImage)res->image,
            .pObjectName = desc.name.c_str(),
        };
        GetState().vkSetDebugUtilsObjectNameEXT(GetState().device, &name);
#endif

        InitializeImageView(res);

        // Alloc descriptor index
        {
            res->resourceid = S.imageSlots.alloc();

            // Bind for Sampler
            if ((int)desc.usage & (int)ImageUsage::Sampled) {
                VkDescriptorImageInfo info = {};
                info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                info.imageView = res->view;
                info.sampler = res->sampler;

                VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                write.pImageInfo = &info;
                write.descriptorCount = 1;
                write.dstSet = S.descriptorSet;
                write.dstBinding = BINDING_SAMPLER;
                write.dstArrayElement = res->resourceid;
                vkUpdateDescriptorSets(S.device, 1, &write, 0, nullptr);
            }

            // Bind for Storage
            if ((int)usage & (int)ImageUsage::Storage) {
                VkDescriptorImageInfo info = {};
                info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                info.imageView = res->view;
                info.sampler = res->sampler;

                VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                write.pImageInfo = &info;
                write.descriptorCount = 1;
                write.dstSet = S.descriptorSet;
                write.dstBinding = BINDING_IMAGE;
                write.dstArrayElement = res->resourceid;
                vkUpdateDescriptorSets(S.device, 1, &write, 0, nullptr);
            }
        }

        return Image(res);
    }

    ///////////////
    // Pipelines //
    ///////////////
    VkShaderModule createShader(const std::vector<uint8_t>& spirv) {
        VkShaderModuleCreateInfo shaderci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shaderci.pCode = (uint32_t*)spirv.data();
        shaderci.codeSize = (spirv.end() - spirv.begin());
        VkShaderModule mod;
        CHECK_VK(vkCreateShaderModule(GetState().device, &shaderci, nullptr, &mod));
        return mod;
    }
    VkPipeline createGraphicsPipeline(const PipelineDesc& desc, VkShaderModule vertexShader, VkShaderModule fragmentShader) {
        auto& S = GetState();

        std::vector<VkVertexInputAttributeDescription> attributes;
        std::vector<VkVertexInputBindingDescription> bindings;

        for (int bindingIdx = 0; bindingIdx < desc.bindings.size(); bindingIdx++) {
            uint32_t attributesStride = 0;
            uint32_t attributeIndex = 0;
            for (int attributeIdx = 0; attributeIdx < desc.bindings[bindingIdx].size(); attributeIdx++) {
                Format fmt = desc.bindings[bindingIdx][attributeIdx];

                VkVertexInputAttributeDescription attrib = {};
                attrib.binding = bindingIdx;
                attrib.location = attributeIndex;
                attrib.offset = attributesStride;
                attrib.format = FORMAT_VK[(int)fmt].format;
                attributes.push_back(attrib);

                attributesStride += (uint32_t)FORMAT_VK[(int)fmt].size;
                attributeIndex++;
            }
            if (attributeIndex > 0) {
                VkVertexInputBindingDescription binding = {};
                binding.binding = bindingIdx;
                binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
                binding.stride = attributesStride;
                bindings.push_back(binding);
            }
        }

        VkViewport viewport = {0.0f, 0.0f, 1024.0f, 720.0f, 0.0f, 1.0f};
        VkRect2D scissor = {{0, 0}, {std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::max()}};
        std::vector<VkPipelineColorBlendAttachmentState> attachments;
        std::vector<VkDynamicState> dynamicState = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_LINE_WIDTH};

        VkPipelineShaderStageCreateInfo stages[2];
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stages[0].module = vertexShader;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].pName = "main";
        stages[0].pSpecializationInfo = nullptr;
        stages[0].flags = 0;
        stages[0].pNext = nullptr;
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stages[1].module = fragmentShader;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].pName = "main";
        stages[1].pSpecializationInfo = nullptr;
        stages[1].flags = 0;
        stages[1].pNext = nullptr;

        VkPipelineVertexInputStateCreateInfo vertexInfo = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vertexInfo.vertexBindingDescriptionCount = (uint32_t)bindings.size();
        vertexInfo.pVertexBindingDescriptions = bindings.data();
        vertexInfo.vertexAttributeDescriptionCount = (uint32_t)attributes.size();
        vertexInfo.pVertexAttributeDescriptions = attributes.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        inputAssemblyInfo.topology = PRIMITIVE_TOPOLOGY_VK[(int)desc.primitive];

        VkPipelineViewportStateCreateInfo viewportInfo = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewportInfo.viewportCount = 1;
        viewportInfo.pViewports = &viewport;
        viewportInfo.scissorCount = 1;
        viewportInfo.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizationInfo = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rasterizationInfo.polygonMode = desc.wireframe ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
        rasterizationInfo.cullMode = TO_VK_CULL_MODE[(int)desc.cull];
        rasterizationInfo.frontFace = desc.frontClockwise ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizationInfo.lineWidth = 1.0f;
        rasterizationInfo.depthClampEnable = false;
        rasterizationInfo.depthBiasEnable = false;
        rasterizationInfo.rasterizerDiscardEnable = false;

        VkPipelineMultisampleStateCreateInfo multisampleInfo{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisampleInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencilInfo = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depthStencilInfo.depthTestEnable = desc.depthTest;
        depthStencilInfo.depthWriteEnable = desc.depthWrite;
        depthStencilInfo.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencilInfo.depthBoundsTestEnable = true;
        depthStencilInfo.minDepthBounds = 0.0f;
        depthStencilInfo.maxDepthBounds = 1.0f;
        depthStencilInfo.stencilTestEnable = false;
        depthStencilInfo.front = {};
        depthStencilInfo.back = {};

        int renderingColorAttachmentCount = 0;
        VkFormat renderingFormats[MAX_ATTACHMENTS_COUNT];
        VkFormat renderingDepthStencilFormat = VK_FORMAT_UNDEFINED;
        VkPipelineColorBlendStateCreateInfo colorBlendInfo = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};

        EVK_ASSERT(desc.attachments.size() < MAX_ATTACHMENTS_COUNT, "Attachments count is bigger than max %d", MAX_ATTACHMENTS_COUNT);

        for (int i = 0; i < desc.attachments.size(); i++) {
            Blend blend = (i < desc.blends.size()) ? desc.blends[i] : Blend::Disabled;

            if (DoesFormatHaveDepth(desc.attachments[i])) {
                EVK_ASSERT(renderingDepthStencilFormat == VK_FORMAT_UNDEFINED, "Only one depth stencil attachment is allowed");
                EVK_ASSERT(i == desc.attachments.size() - 1, "Depth stencil attachment must be specified at the end");
                renderingDepthStencilFormat = FORMAT_VK[(int)desc.attachments[i]].format;
                break;
            }

            VkPipelineColorBlendAttachmentState blendAttach = {};
            blendAttach.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            switch (blend) {
                case Blend::Disabled:
                    blendAttach.blendEnable = false;
                    break;
                case Blend::Alpha:
                    blendAttach.blendEnable = true;
                    blendAttach.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
                    blendAttach.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                    blendAttach.colorBlendOp = VK_BLEND_OP_ADD;
                    blendAttach.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
                    blendAttach.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
                    blendAttach.alphaBlendOp = VK_BLEND_OP_ADD;
                    break;
                case Blend::Additive:
                    blendAttach.blendEnable = true;
                    blendAttach.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
                    blendAttach.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
                    blendAttach.colorBlendOp = VK_BLEND_OP_ADD;
                    blendAttach.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
                    blendAttach.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
                    blendAttach.alphaBlendOp = VK_BLEND_OP_ADD;
                    break;
                default:
                    break;
            }

            renderingFormats[renderingColorAttachmentCount] = FORMAT_VK[(int)desc.attachments[i]].format;
            renderingColorAttachmentCount++;
            attachments.push_back(blendAttach);
        }
        colorBlendInfo.attachmentCount = (uint32_t)attachments.size();
        colorBlendInfo.pAttachments = attachments.data();

        VkPipelineDynamicStateCreateInfo dynamicStateInfo = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamicStateInfo.dynamicStateCount = (uint32_t)dynamicState.size();
        dynamicStateInfo.pDynamicStates = dynamicState.data();

        VkPipelineRenderingCreateInfoKHR renderingCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
        renderingCreateInfo.colorAttachmentCount = renderingColorAttachmentCount;
        renderingCreateInfo.pColorAttachmentFormats = renderingFormats;
        renderingCreateInfo.depthAttachmentFormat = renderingDepthStencilFormat;
        renderingCreateInfo.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
        if (DoesFormatHaveStencil(desc.attachments.back())) {
            renderingCreateInfo.stencilAttachmentFormat = renderingDepthStencilFormat;
        }
        renderingCreateInfo.viewMask = 0;
        renderingCreateInfo.pNext = nullptr;

        VkGraphicsPipelineCreateInfo createInfo = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        createInfo.stageCount = 2;
        createInfo.pStages = stages;
        createInfo.pVertexInputState = &vertexInfo;
        createInfo.pInputAssemblyState = &inputAssemblyInfo;
        createInfo.pViewportState = &viewportInfo;
        createInfo.pRasterizationState = &rasterizationInfo;
        createInfo.pMultisampleState = &multisampleInfo;
        createInfo.pDepthStencilState = &depthStencilInfo;
        createInfo.pColorBlendState = &colorBlendInfo;
        createInfo.pDynamicState = &dynamicStateInfo;
        createInfo.layout = S.pipelineLayout;
        createInfo.renderPass = VK_NULL_HANDLE;
        createInfo.pNext = &renderingCreateInfo;

        VkPipeline pipeline = VK_NULL_HANDLE;
        vkCreateGraphicsPipelines(GetState().device, VK_NULL_HANDLE, 1, &createInfo, nullptr, &pipeline);

        return pipeline;
    }
    VkPipeline createComputePipeline(const PipelineDesc& desc, VkShaderModule computeShader) {
        VkComputePipelineCreateInfo pipelineci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pipelineci.stage.module = computeShader;
        pipelineci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineci.stage.pName = "main";
        pipelineci.layout = GetState().pipelineLayout;

        VkPipeline pipeline = VK_NULL_HANDLE;
        CHECK_VK(vkCreateComputePipelines(GetState().device, nullptr, 1, &pipelineci, nullptr, &pipeline));
        return pipeline;
    }
    Pipeline CreatePipeline(const PipelineDesc& desc) {
        Internal_Pipeline* state = new Internal_Pipeline();
        state->desc = desc;

        if (!desc.VS.empty() && !desc.FS.empty()) {
            VkShaderModule v = createShader(desc.VS);
            VkShaderModule f = createShader(desc.FS);
            EVK_ASSERT(v != VK_NULL_HANDLE, "Failed to compile vertex shader");
            EVK_ASSERT(f != VK_NULL_HANDLE, "Failed to compile fragment shader");

            state->pipeline = createGraphicsPipeline(desc, v, f);
            state->isCompute = false;

            vkDestroyShaderModule(GetState().device, v, nullptr);
            vkDestroyShaderModule(GetState().device, f, nullptr);
        } else if (!desc.CS.empty()) {
            VkShaderModule k = createShader(desc.CS);
            EVK_ASSERT(k != VK_NULL_HANDLE, "Failed to compile compute shader");

            state->pipeline = createComputePipeline(desc, k);
            state->isCompute = true;

            vkDestroyShaderModule(GetState().device, k, nullptr);
        } else {
            EVK_ASSERT(false, "Spirv not provided!");
        }

        EVK_ASSERT(state->pipeline != VK_NULL_HANDLE, "Failed to create pipeline '%s'", desc.name.c_str());

#if EVK_DEBUG
        VkDebugUtilsObjectNameInfoEXT name = {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .objectType = VkObjectType::VK_OBJECT_TYPE_PIPELINE,
            .objectHandle = (uint64_t)state->pipeline,
            .pObjectName = desc.name.c_str(),
        };
        GetState().vkSetDebugUtilsObjectNameEXT(GetState().device, &name);
#endif

        return Pipeline{state};
    }

    const BufferDesc& GetDesc(const Buffer& res) {
        return ToInternal(res).desc;
    }
    const ImageDesc& GetDesc(const Image& res) {
        return ToInternal(res).desc;
    }

    FrameData::~FrameData() {
        auto& D = GetState().device;
        vkDestroySemaphore(D, imageReadySemaphore, nullptr);
        vkDestroyCommandPool(D, pool, nullptr);
        vkDestroySemaphore(D, cmdDoneSemaphore, nullptr);
        vkDestroyFence(D, fence, nullptr);
        vkDestroyQueryPool(D, queryPool, nullptr);
    }
    State& GetState() {
        EVK_ASSERT(GState, "EVK not intialized! did you call evk::Initialize()?");
        return *GState;
    }
    void SetState(State* state) {
        EVK_ASSERT(state != nullptr, "State is null!");
        EVK_ASSERT(GState == nullptr, "State already initialized!");
        GState = state;
    }
    FrameData& GetFrame() {
        return GetState().frames[GetState().frame];
    }
    void _BeginFrame() {
        auto& S = GetState();
        auto& F = GetFrame();

        F.timestampNames.clear();
        F.doingPresent = false;
        F.insideRenderPass = false;
        F.stagingOffset = 0;
        CHECK_VK(vkResetCommandPool(S.device, F.pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT));

        VkCommandBufferBeginInfo cmdbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        cmdbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK_VK(vkBeginCommandBuffer(F.cmd, &cmdbi));

        vkCmdResetQueryPool(F.cmd, F.queryPool, 0, PERF_QUERY_COUNT);
        vkCmdBindDescriptorSets(F.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, S.pipelineLayout, 0, 1, &S.descriptorSet, 0, nullptr);
        vkCmdBindDescriptorSets(F.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, S.pipelineLayout, 0, 1, &S.descriptorSet, 0, nullptr);
    }
    void _EndFrame() {
        auto& S = GetState();
        auto& F = GetFrame();

        CHECK_VK(vkEndCommandBuffer(F.cmd));

        VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.waitSemaphoreCount = F.doingPresent ? 1 : 0;
        submit.pWaitSemaphores = F.doingPresent ? &S.frames[S.swapchainSemaphoreIndex].imageReadySemaphore : nullptr;
        submit.signalSemaphoreCount = F.doingPresent ? 1 : 0;
        submit.pSignalSemaphores = F.doingPresent ? &F.cmdDoneSemaphore : nullptr;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &F.cmd;
        submit.pWaitDstStageMask = &dstStage;
        CHECK_VK(vkQueueSubmit(S.queue, 1, &submit, F.fence));

        if (F.doingPresent) {
            VkPresentInfoKHR present = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
            present.waitSemaphoreCount = 1;
            present.pWaitSemaphores = &F.cmdDoneSemaphore;
            present.swapchainCount = 1;
            present.pSwapchains = &S.swapchain;
            present.pImageIndices = &S.swapchainIndex;
            VkResult r = vkQueuePresentKHR(S.queue, &present);

            if (r == VK_ERROR_OUT_OF_DATE_KHR) {
                S.frame = (int)S.frames.size() - 1;
                RecreateSwapchain();
            }
        }
    }
    void _WaitFrameCompletion() {
        auto& F = GetFrame();

        CHECK_VK(vkWaitForFences(GetState().device, 1, &F.fence, true, std::numeric_limits<uint64_t>().max()));
        CHECK_VK(vkResetFences(GetState().device, 1, &F.fence));
        F.queries.resize(PERF_QUERY_COUNT);

        if(F.timestampNames.empty() == false) { 
            vkGetQueryPoolResults(GetState().device, F.queryPool, 0, PERF_QUERY_COUNT, PERF_QUERY_COUNT * sizeof(uint64_t), F.queries.data(), 8, VK_QUERY_RESULT_64_BIT);
            F.timestampEntries.clear();
            uint64_t start = F.queries[0];
            for (int i = 0; i < F.timestampNames.size(); i++) {
                TimestampEntry e = {};
                e.start = (F.queries[i * 2] - start) * 1e-6 / GetState().timestampPeriod;
                e.end = (F.queries[i * 2 + 1] - start) * 1e-6 / GetState().timestampPeriod;
                e.name = F.timestampNames[i];
                F.timestampEntries.push_back(e);
            }
        }
    }
    void _ReleaseResources() {
        auto& F = GetFrame();

        for (int i = 0; i < F.toDelete.size(); i++) {
            delete F.toDelete[i];
        }
        F.toDelete.clear();
    }

    //////////////////////
    // Global Functions //
    //////////////////////
    bool InitializeEVK(const EvkDesc& desc) {
        EVK_ASSERT(!GState, "EVK already initialized!");
        GState = new State();
        State& S = GetState();

        // Application and Instance
        {
            // Application Info
            VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
            appInfo.apiVersion = VK_API_VERSION_1_3;
            appInfo.pApplicationName = desc.applicationName.c_str();
            appInfo.applicationVersion = desc.applicationVersion;
            appInfo.pEngineName = desc.engineName.c_str();
            appInfo.engineVersion = desc.engineVersion;

            
            std::vector<std::string> instanceLayers = {};
            std::vector<std::string> instanceExtensions = {};

            if (desc.enableSwapchain) {
                instanceExtensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
                #if WIN32
                    instanceExtensions.push_back("VK_KHR_win32_surface");
                #else
                    #error "Unsupported platform"
                #endif
            }

            // Instance Info
            std::vector<const char*> extensions;
            for (auto& name : instanceExtensions) {
                extensions.push_back(name.c_str());
            }
            std::vector<const char*> layers;
            for (auto& name : instanceLayers) {
                layers.push_back(name.c_str());
            }

#if EVK_DEBUG
            printf("[evk] validation layers enabled!\n");
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            layers.push_back("VK_LAYER_KHRONOS_validation");
#endif

            // Validate instance layers
            {
                uint32_t count = 0;
                std::vector<VkLayerProperties> avaible_layers;
                CHECK_VK(vkEnumerateInstanceLayerProperties(&count, nullptr));
                avaible_layers.resize(count);
                CHECK_VK(vkEnumerateInstanceLayerProperties(&count, avaible_layers.data()));

                for (auto& layer : layers) {
                    bool found = false;
                    for (auto& avaible_layer : avaible_layers) {
                        if (strcmp(layer, avaible_layer.layerName) == 0) {
                            found = true;
                            break;
                        }
                    }
                    EVK_ASSERT(found, "Failed to find instance layer '%s'!", layer);
                }
            }
            // Validate instance extensions
            {
                uint32_t count = 0;
                std::vector<VkExtensionProperties> avaible_extensions;
                CHECK_VK(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr));
                avaible_extensions.resize(count);
                CHECK_VK(vkEnumerateInstanceExtensionProperties(nullptr, &count, avaible_extensions.data()));

                for (auto& extension : extensions) {
                    bool found = false;
                    for (auto& avaible_extension : avaible_extensions) {
                        if (strcmp(extension, avaible_extension.extensionName) == 0) {
                            found = true;
                            break;
                        }
                    }
                    EVK_ASSERT(found, "Failed to find instance extension '%s'!", extension);
                }
            }

            VkInstanceCreateInfo instanceci = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
            instanceci.pApplicationInfo = &appInfo;
            instanceci.enabledExtensionCount = uint32_t(extensions.size());
            instanceci.ppEnabledExtensionNames = extensions.data();
            instanceci.enabledLayerCount = uint32_t(layers.size());
            instanceci.ppEnabledLayerNames = layers.data();

            // #if EVK_DEBUG
            //     VkValidationFeatureEnableEXT validationEnable[] = { VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT };
            //     VkValidationFeaturesEXT feature_validation = {VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
            //     feature_validation.disabledValidationFeatureCount = 0;
            //     feature_validation.pDisabledValidationFeatures = nullptr;
            //     feature_validation.enabledValidationFeatureCount = 1;
            //     feature_validation.pEnabledValidationFeatures = validationEnable;
            //     feature_validation.pNext = instanceci.pNext;
            //     instanceci.pNext = &feature_validation;
            // #endif

            CHECK_VK(vkCreateInstance(&instanceci, nullptr, &S.instance));
        }

        // Device and Queues
        {
            // PhysicalDevice
            uint32_t physicalDeviceCount = 0;
            std::vector<VkPhysicalDevice> physicalDevices;
            CHECK_VK(vkEnumeratePhysicalDevices(S.instance, &physicalDeviceCount, nullptr));
            physicalDevices.resize(physicalDeviceCount);
            CHECK_VK(vkEnumeratePhysicalDevices(S.instance, &physicalDeviceCount, physicalDevices.data()));
            EVK_ASSERT(physicalDeviceCount > 0, "could not find physical device!");
            S.physicalDevice = physicalDevices[0];

            // PhysicalDevice properties
            VkPhysicalDeviceProperties props = {};
            vkGetPhysicalDeviceProperties(S.physicalDevice, &props);
            EVK_ASSERT(props.limits.timestampComputeAndGraphics, "Timestamp not supported!");

            S.timestampPeriod = props.limits.timestampPeriod;

            printf("[evk] Vulkan %d.%d.%d | %s \n", VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion), VK_API_VERSION_PATCH(props.apiVersion), props.deviceName);

            uint32_t familyPropsCount = 0;
            std::vector<VkQueueFamilyProperties> familyProps;
            vkGetPhysicalDeviceQueueFamilyProperties(S.physicalDevice, &familyPropsCount, nullptr);
            familyProps.resize(familyPropsCount);
            vkGetPhysicalDeviceQueueFamilyProperties(S.physicalDevice, &familyPropsCount, familyProps.data());

            S.queueFamily = 0;
            for (const auto& queueFamily : familyProps) {
                if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                    break;
                }
                S.queueFamily++;
            }

            float priority0 = 1.0f;
            VkDeviceQueueCreateInfo deviceQueueci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            deviceQueueci.queueFamilyIndex = S.queueFamily;
            deviceQueueci.queueCount = 1;
            deviceQueueci.pQueuePriorities = &priority0;

            // Device features
            VkPhysicalDeviceFeatures features = {};
            features.geometryShader = true;
            features.shaderInt64 = true;
            features.fillModeNonSolid = true;
            features.shaderStorageImageReadWithoutFormat = true;
            features.shaderStorageImageWriteWithoutFormat = true;
            features.independentBlend = true;
            features.wideLines = true;
            
            VkPhysicalDeviceShaderAtomicFloatFeaturesEXT feature_atomicFloat = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
                .shaderBufferFloat32Atomics = true,
                .shaderBufferFloat32AtomicAdd = true,
            };
            VkPhysicalDeviceShaderAtomicInt64Features feature_atomicint64 = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES,
                .pNext = &feature_atomicFloat,
                .shaderBufferInt64Atomics = true,
            };
            VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT feature_imageatomicint64 = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT,
                .pNext = &feature_atomicint64,
                .shaderImageInt64Atomics = true,
            };
            // Feature Descriptor Indexing Feature
            VkPhysicalDeviceDescriptorIndexingFeatures feature_indexing = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
                .pNext = &feature_imageatomicint64,
                .shaderSampledImageArrayNonUniformIndexing = true,
                .shaderStorageBufferArrayNonUniformIndexing = true,
                .shaderStorageImageArrayNonUniformIndexing = true,
                .descriptorBindingSampledImageUpdateAfterBind = true,
                .descriptorBindingStorageImageUpdateAfterBind = true,
                .descriptorBindingStorageBufferUpdateAfterBind = true,
                .descriptorBindingUpdateUnusedWhilePending = true,
                .descriptorBindingPartiallyBound = true,
                .runtimeDescriptorArray = true,
            };
            VkPhysicalDeviceDynamicRenderingFeatures feature_dynamicRendering = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
                .pNext = &feature_indexing,
                .dynamicRendering = true,
            };
            VkPhysicalDeviceBufferDeviceAddressFeatures feature_bufferDeviceAddress = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
                .pNext = &feature_dynamicRendering,
                .bufferDeviceAddress = true,
            };
            VkPhysicalDeviceSynchronization2Features feature_sync2 = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
                .pNext = &feature_bufferDeviceAddress,
                .synchronization2 = true,
            };

            VkPhysicalDevice8BitStorageFeatures feature_8bitStorage = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
                .pNext = &feature_sync2,
                .storageBuffer8BitAccess = true,
                .uniformAndStorageBuffer8BitAccess = true,
            };
            VkPhysicalDevice16BitStorageFeatures feature_16bitStorage = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
                .pNext = &feature_8bitStorage,
                .storageBuffer16BitAccess = true,
                .uniformAndStorageBuffer16BitAccess = true,
            };
            VkPhysicalDeviceRayTracingPipelineFeaturesKHR feature_rayTracingPipeline = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
                .pNext = &feature_16bitStorage,
                .rayTracingPipeline = VK_TRUE,
            };
            VkPhysicalDeviceAccelerationStructureFeaturesKHR feature_accelerationStructure = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
                .pNext = &feature_rayTracingPipeline,
                .accelerationStructure = VK_TRUE,
                .accelerationStructureCaptureReplay = VK_TRUE,
                .descriptorBindingAccelerationStructureUpdateAfterBind = VK_TRUE,
            };
            VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR feature_rtPositionFetch = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR,
                .pNext = &feature_accelerationStructure,
                .rayTracingPositionFetch = true,
            };
            VkPhysicalDeviceRayQueryFeaturesKHR feature_rayQuery = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
                .pNext = &feature_rtPositionFetch,
                .rayQuery = VK_TRUE,
            };


            // Create device and get queue
            std::vector<const char*> deviceExtensions = {
                VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME,
                VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,

                VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
                VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
                VK_KHR_RAY_QUERY_EXTENSION_NAME,
                VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME,
            };

// Check device extensions support
#if EVK_DEBUG
            std::vector<VkExtensionProperties> supportedExtensions;
            uint32_t supportedExtensionsCount = 0;
            CHECK_VK(vkEnumerateDeviceExtensionProperties(S.physicalDevice, NULL, &supportedExtensionsCount, NULL));
            supportedExtensions.resize(supportedExtensionsCount);
            CHECK_VK(vkEnumerateDeviceExtensionProperties(S.physicalDevice, NULL, &supportedExtensionsCount, supportedExtensions.data()));
            for (auto& ext1 : deviceExtensions) {
                bool found = false;
                for (auto& ext2 : supportedExtensions) {
                    if (!std::strcmp(ext1, ext2.extensionName)) {
                        found = true;
                        break;
                    }
                }
                EVK_ASSERT(found, "Device extension '%s' not found!", ext1);
            }
#endif

            VkDeviceCreateInfo deviceci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
            deviceci.pEnabledFeatures = &features;
            deviceci.queueCreateInfoCount = 1;
            deviceci.pQueueCreateInfos = &deviceQueueci;
            deviceci.enabledExtensionCount = (uint32_t)deviceExtensions.size();
            deviceci.ppEnabledExtensionNames = deviceExtensions.data();
            deviceci.pNext = &feature_rayQuery;


            CHECK_VK(vkCreateDevice(S.physicalDevice, &deviceci, nullptr, &S.device));
            vkGetDeviceQueue(S.device, S.queueFamily, 0, &S.queue);
        }

        // Get PFNs
        { 
            #define EVK_PFN(name) S.name = (PFN_##name)vkGetDeviceProcAddr(S.device, #name)
            EVK_PFN(vkCmdBeginDebugUtilsLabelEXT);
            EVK_PFN(vkCmdEndDebugUtilsLabelEXT);
            EVK_PFN(vkSetDebugUtilsObjectNameEXT);
        }

        // Vma Allocator
        {
            VmaAllocatorCreateInfo allocatorInfo = {};
            allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
            allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
            allocatorInfo.instance = S.instance;
            allocatorInfo.physicalDevice = S.physicalDevice;
            allocatorInfo.device = S.device;
            vmaCreateAllocator(&allocatorInfo, &S.allocator);
        }

        // Descriptors
        {
            // layout(binding = 0) buffer //
            // layout(binding = 1) sampler //
            // layout(binding = 2) image //

            // Create descriptor pool
            std::vector<VkDescriptorPoolSize> poolSizes = {
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, STORAGE_COUNT},
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, SAMPLER_COUNT},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, IMAGE_COUNT},
                {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, TLAS_COUNT},
            };
            VkDescriptorPoolCreateInfo poolci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            poolci.poolSizeCount = (uint32_t)poolSizes.size();
            poolci.pPoolSizes = poolSizes.data();
            poolci.maxSets = 1;
            poolci.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
            CHECK_VK(vkCreateDescriptorPool(S.device, &poolci, nullptr, &S.descriptorPool));

            // Create layout set
            std::vector<VkDescriptorSetLayoutBinding> bindings = {
                {BINDING_STORAGE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, STORAGE_COUNT, VK_SHADER_STAGE_ALL},
                {BINDING_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, SAMPLER_COUNT, VK_SHADER_STAGE_ALL},
                {BINDING_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, IMAGE_COUNT, VK_SHADER_STAGE_ALL},
                {BINDING_TLAS, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, TLAS_COUNT, VK_SHADER_STAGE_ALL},
            };

            // Flag each binding as partially bound and update after bind
            VkDescriptorSetLayoutBindingFlagsCreateInfo setLayoutBindingsFlags = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
            std::vector<VkDescriptorBindingFlags> bindingFlags = {
                VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
                VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
                VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
                VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
            };
            setLayoutBindingsFlags.bindingCount = (uint32_t)bindingFlags.size();
            setLayoutBindingsFlags.pBindingFlags = bindingFlags.data();

            VkDescriptorSetLayoutCreateInfo setLayoutci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            setLayoutci.bindingCount = (uint32_t)bindings.size();
            setLayoutci.pBindings = bindings.data();
            setLayoutci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
            setLayoutci.pNext = &setLayoutBindingsFlags;
            CHECK_VK(vkCreateDescriptorSetLayout(S.device, &setLayoutci, nullptr, &S.descriptorSetLayout));

            VkPushConstantRange pushc = {};
            pushc.offset = 0;
            pushc.size = 128;
            pushc.stageFlags = VK_SHADER_STAGE_ALL;

            VkPipelineLayoutCreateInfo layoutci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layoutci.setLayoutCount = 1;
            layoutci.pSetLayouts = &S.descriptorSetLayout;
            layoutci.pushConstantRangeCount = 1;
            layoutci.pPushConstantRanges = &pushc;
            CHECK_VK(vkCreatePipelineLayout(S.device, &layoutci, nullptr, &S.pipelineLayout));

            // Allocate single global descriptor set
            VkDescriptorSetAllocateInfo descset = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            descset.descriptorPool = S.descriptorPool;
            descset.descriptorSetCount = 1;
            descset.pSetLayouts = &S.descriptorSetLayout;
            CHECK_VK(vkAllocateDescriptorSets(S.device, &descset, &S.descriptorSet));
        }

        // Frames data
        {
            S.frames.resize(desc.frameBufferingCount);
            for (uint32_t i = 0; i < desc.frameBufferingCount; i++) {
                FrameData& fd = S.frames[i];
                VkSemaphoreCreateInfo semaphoreci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
                CHECK_VK(vkCreateSemaphore(S.device, &semaphoreci, nullptr, &fd.imageReadySemaphore));

                VkCommandPoolCreateInfo cmdPoolci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
                cmdPoolci.queueFamilyIndex = S.queueFamily;
                CHECK_VK(vkCreateCommandPool(S.device, &cmdPoolci, nullptr, &fd.pool));

                VkCommandBufferAllocateInfo cmdAlloc = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
                cmdAlloc.commandBufferCount = 1;
                cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                cmdAlloc.commandPool = fd.pool;
                CHECK_VK(vkAllocateCommandBuffers(S.device, &cmdAlloc, &fd.cmd));

                VkFenceCreateInfo fenceci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
                fenceci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
                CHECK_VK(vkCreateFence(S.device, &fenceci, nullptr, &fd.fence));

                VkQueryPoolCreateInfo queryPoolci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
                queryPoolci.queryCount = PERF_QUERY_COUNT;
                queryPoolci.queryType = VK_QUERY_TYPE_TIMESTAMP;
                CHECK_VK(vkCreateQueryPool(S.device, &queryPoolci, nullptr, &fd.queryPool));

                CHECK_VK(vkCreateSemaphore(S.device, &semaphoreci, nullptr, &fd.cmdDoneSemaphore));

                fd.stagingBuffer = CreateBuffer({
                    .name = "Staging buffer",
                    .size = 64'000'000,
                    .usage = BufferUsage::TransferSrc,
                    .memoryType = MemoryType::CPU_TO_GPU,
                });
            }
        }

        // Raytracing
        {
            VkDevice device = GetState().device;
            GetState().vkGetAccelerationStructureBuildSizesKHR = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR");
            GetState().vkCreateAccelerationStructureKHR = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR");
            GetState().vkGetBufferDeviceAddressKHR = (PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddressKHR");
            GetState().vkCmdBuildAccelerationStructuresKHR = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR");
            GetState().vkGetAccelerationStructureDeviceAddressKHR = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR");
            GetState().vkCreateRayTracingPipelinesKHR = (PFN_vkCreateRayTracingPipelinesKHR)vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesKHR");
            GetState().vkGetRayTracingShaderGroupHandlesKHR = (PFN_vkGetRayTracingShaderGroupHandlesKHR)vkGetDeviceProcAddr(device, "vkGetRayTracingShaderGroupHandlesKHR");
            GetState().vkCmdTraceRaysKHR = (PFN_vkCmdTraceRaysKHR)vkGetDeviceProcAddr(device, "vkCmdTraceRaysKHR");
            GetState().vkDestroyAccelerationStructureKHR = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR");
        }

        _WaitFrameCompletion();
        _BeginFrame();

        return true;
    }
    void Shutdown() {
        auto& S = GetState();

        // release internal resources
        {
            for (auto& f : S.frames) {
                f.image.release();
            }
        }

        _EndFrame();
        _WaitFrameCompletion();
        _ReleaseResources();

        S.frames.clear();

        vmaDestroyAllocator(GetState().allocator);
        vkDestroyDescriptorPool(S.device, S.descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(S.device, S.descriptorSetLayout, nullptr);
        vkDestroyPipelineLayout(S.device, S.pipelineLayout, nullptr);
        vkDestroySwapchainKHR(S.device, S.swapchain, nullptr);
        vkDestroyDevice(S.device, nullptr);
        vkDestroySurfaceKHR(S.instance, S.surface, nullptr);
        vkDestroyInstance(S.instance, nullptr);

        delete GState;
        GState = nullptr;
    }

    ///////////////
    // Swapchain //
    ///////////////
    Image CreateImageForSwapchain(VkImage image, uint32_t width, uint32_t height) {
        auto& S = GetState();
        Internal_Image* res = new Internal_Image();

        res->desc.name = "Swapchain color image";
        res->desc.format = Format::BGRA8Unorm;
        res->desc.usage = ImageUsage::Attachment;
        res->desc.extent = {width, height};
        res->desc.filter = Filter::Linear;
        res->image = image;

        InitializeImageView(res);

        return Image(res);
    }
    bool RecreateSwapchain() {
        auto& S = GetState();
        CHECK_VK(vkDeviceWaitIdle(S.device));
        EVK_ASSERT(S.surface != nullptr, "Surface is not initialized!");

        auto oldSwapchain = S.swapchain;
        VkSurfaceKHR surface = (VkSurfaceKHR)S.surface;

        VkBool32 supported = false;
        CHECK_VK(vkGetPhysicalDeviceSurfaceSupportKHR(S.physicalDevice, S.queueFamily, surface, &supported));
        if (!supported) {
            return false;
        }

        // Check if surface supports format and colorSpace
        {
            bool foundSurfaceFormat = false;
            uint32_t formatCount = 0;
            std::vector<VkSurfaceFormatKHR> surfaceFormats;
            CHECK_VK(vkGetPhysicalDeviceSurfaceFormatsKHR(S.physicalDevice, S.surface, &formatCount, nullptr));
            surfaceFormats.resize(formatCount);
            CHECK_VK(vkGetPhysicalDeviceSurfaceFormatsKHR(S.physicalDevice, S.surface, &formatCount, surfaceFormats.data()));

            for (auto surfaceFormat : surfaceFormats) {
                // TODO: Do a proper format detection for swapchain
                if (surfaceFormat.format == VK_FORMAT_B8G8R8A8_UNORM && surfaceFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                    foundSurfaceFormat = true;
                }
            }
            EVK_ASSERT(foundSurfaceFormat, "Failed to find surface format!");
        }

        VkSurfaceCapabilitiesKHR surfaceCaps;
        CHECK_VK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(S.physicalDevice, S.surface, &surfaceCaps));
        // auto capabilities = S.physicalDevice.getSurfaceCapabilitiesKHR(surface);i
        VkFormat format = VK_FORMAT_B8G8R8A8_UNORM;
        VkColorSpaceKHR colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        auto frameCount = S.frames.size();
        VkSurfaceTransformFlagBitsKHR transform = surfaceCaps.currentTransform;
        VkExtent2D extent = surfaceCaps.currentExtent;

        EVK_ASSERT(surfaceCaps.minImageCount <= S.frames.size() && S.frames.size() <= surfaceCaps.maxImageCount, "Frame buffering count out of range!");

        VkSwapchainCreateInfoKHR swapchainci = {
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = surface,
            .minImageCount = (uint32_t)frameCount,
            .imageFormat = format,
            .imageColorSpace = colorSpace,
            .imageExtent = extent,
            .imageArrayLayers = 1,
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .preTransform = transform,
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = VK_PRESENT_MODE_FIFO_KHR,  // VK_PRESENT_MODE_MAILBOX_KHR = v-sync off,
            .clipped = false,
            .oldSwapchain = oldSwapchain,
        };

        CHECK_VK(vkCreateSwapchainKHR(S.device, &swapchainci, nullptr, &S.swapchain));
        S.swapchainIndex = (uint32_t)(frameCount - 1);

        uint32_t imageCount = 0;
        std::vector<VkImage> images;
        CHECK_VK(vkGetSwapchainImagesKHR(S.device, S.swapchain, &imageCount, nullptr));
        images.resize(imageCount);
        CHECK_VK(vkGetSwapchainImagesKHR(S.device, S.swapchain, &imageCount, images.data()));
        for (int i = 0; i < frameCount; i++) {
            FrameData& d = S.frames[i];
            d.image = CreateImageForSwapchain(images[i], extent.width, extent.height);
        }

        vkDestroySwapchainKHR(S.device, oldSwapchain, nullptr);

        return true;
    }
    bool InitializeSwapchain(void* vulkanSurfaceKHR) {
        GetState().surface = (VkSurfaceKHR)vulkanSurfaceKHR;
        return RecreateSwapchain();
    }

    void Submit() {
        auto& S = GetState();
        auto& F = GetFrame();

        _EndFrame();
        S.frame_total++;

        // Swap frame
        S.frame = (S.frame + 1) % S.frames.size();

        _WaitFrameCompletion();
        _ReleaseResources();
        _BeginFrame();
    }
    uint32_t GetFrameBufferingCount() {
        return (uint32_t)GetState().frames.size();
    }
    uint32_t GetFrameIndex() {
        return GetState().frame;
    }
    void Sync() {
        for (int i = 0; i < GetState().frames.size(); i++) {
            Submit();
        }
    }
    const std::vector<TimestampEntry>& GetTimestamps() {
        return GetFrame().timestampEntries;
    }

    MemoryBudget GetMemoryBudget() {
        static_assert(sizeof(MemoryBudget::Heap) == sizeof(VmaBudget));
        static_assert(MemoryBudget::MAX_HEAPS == VK_MAX_MEMORY_HEAPS);
        MemoryBudget budget = {};
        vmaGetHeapBudgets(GetState().allocator, (VmaBudget*)&budget);
        return budget;
    }
    // Cmds //
    //////////
    void CmdBind(Pipeline pipeline) {
        bool isCompute = ToInternal(pipeline).isCompute;
        bool isGraphics = !isCompute;
        EVK_ASSERT(pipeline.res != nullptr, "Null pipeline");
        EVK_ASSERT(!isGraphics || GetFrame().insideRenderPass, "graphics pipeline bind must be inside a render pass.");
        EVK_ASSERT(!isCompute || !GetFrame().insideRenderPass, "compute pipeline bind must be outside a render pass.");
        vkCmdBindPipeline(GetFrame().cmd, isCompute ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS, ToInternal(pipeline).pipeline);
    }
    void CmdDispatch(uint32_t countX, uint32_t countY, uint32_t countZ) {
        vkCmdDispatch(GetFrame().cmd, countX, countY, countZ);
    }
    void CmdBarrier(Image& image, ImageLayout oldLayout, ImageLayout newLayout, uint32_t mip, uint32_t mipCount, uint32_t layer, uint32_t layerCount) {
        EVK_ASSERT(GetFrame().insideRenderPass == false, "can't be used inside a render pass.");
        bool isDepth = DoesFormatHaveDepth(GetDesc(image).format);
        VkImageAspectFlags aspects = {};
        if (isDepth) {
            aspects |= VK_IMAGE_ASPECT_DEPTH_BIT;
            if (DoesFormatHaveStencil(GetDesc(image).format)) {
                aspects |= VK_IMAGE_ASPECT_STENCIL_BIT;
            }
        } else {
            aspects |= VK_IMAGE_ASPECT_COLOR_BIT;
        }

        VkImageLayout oLayout = IMAGE_LAYOUT_VK[(int)oldLayout];
        VkImageLayout nLayout = IMAGE_LAYOUT_VK[(int)newLayout];

        VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.image = ToInternal(image).image;
        barrier.oldLayout = oLayout;
        barrier.newLayout = nLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask = aspects;
        barrier.subresourceRange.layerCount = layerCount;
        barrier.subresourceRange.baseArrayLayer = layer;
        barrier.subresourceRange.levelCount = mipCount;
        barrier.subresourceRange.baseMipLevel = mip;

        VkPipelineStageFlags srcStage;
        VkPipelineStageFlags dstStage;

        {
            /*
            if (oldLayout == ImageLayout::Undefined) {
                srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                barrier.srcAccessMask = VK_ACCESS_NONE_KHR;
            }
            else if (oldLayout == ImageLayout::TransferSrc) {
                srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            }
            else if (oldLayout == ImageLayout::TransferDst) {
                srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            }
            else if (oldLayout == ImageLayout::ShaderRead) {
                srcStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
                barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            }
            else if (oldLayout == ImageLayout::Attachment) {
                srcStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            }
            else if (oldLayout == ImageLayout::General) {
                srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                barrier.srcAccessMask = VK_ACCESS_NONE_KHR;
            }
            else {
                EVK_ASSERT(false, "Unsupported old layout transition!")
            }

            if (newLayout == ImageLayout::Undefined) {
                dstStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                barrier.dstAccessMask = VK_ACCESS_NONE_KHR;
            }
            else if (newLayout == ImageLayout::TransferSrc) {
                dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                barrier.dstAccessMask  = VK_ACCESS_TRANSFER_READ_BIT;
            }
            else if (newLayout == ImageLayout::TransferDst) {
                dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            }
            else if (newLayout == ImageLayout::ShaderRead) {
                dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            }
            else if (newLayout == ImageLayout::General) {
                dstStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
                barrier.dstAccessMask = VK_ACCESS_NONE_KHR;
            }
            else if (newLayout == ImageLayout::Attachment) {
                dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
                barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            }
            else if (newLayout == ImageLayout::Present) {
                dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                barrier.dstAccessMask = VK_ACCESS_NONE_KHR;
            }
            else {
                EVK_ASSERT(false, "Unsupported new layout transition!")
            }
            */
            srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            barrier.srcAccessMask = VK_ACCESS_NONE_KHR;
            barrier.dstAccessMask = VK_ACCESS_NONE_KHR;
        }
        vkCmdPipelineBarrier(GetFrame().cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }
    void CmdBarrier() {
        VkMemoryBarrier2 barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        };
        VkDependencyInfo dependency = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &barrier,
        };
        vkCmdPipelineBarrier2(GetFrame().cmd, &dependency);
    }
    void CmdFill(Buffer dst, uint32_t data, uint64_t size, uint64_t offset) {
        EVK_ASSERT(size > 0, "Size must be bigger than 0");
        EVK_ASSERT(size % 4 == 0, "Trying to fill buffer '%s', but size is %lld which is not a multiple of 4", GetDesc(dst).name.c_str(), size);
        vkCmdFillBuffer(GetFrame().cmd, ToInternal(dst).buffer, offset, size, data);
    }
    void CmdUpdate(Buffer& dst, uint64_t dstOffset, uint64_t size, void* src) {
        EVK_ASSERT(dstOffset % 4 == 0, "Trying to update buffer '%s', but dstOffset is %lld which is not a multiple of 4", GetDesc(dst).name.c_str(), dstOffset);
        EVK_ASSERT(size % 4 == 0, "Trying to update buffer '%s', but size is %lld which is not a multiple of 4", GetDesc(dst).name.c_str(), size);
        EVK_ASSERT(size <= 65536, "Trying to update buffer '%s', but size is %lld which is not smaller than 65536 ", GetDesc(dst).name.c_str(), size);
        vkCmdUpdateBuffer(GetFrame().cmd, ToInternal(dst).buffer, dstOffset, size, src);
    }
    void CmdBlit(Image& src, Image& dst, ImageRegion srcRegion, ImageRegion dstRegion, Filter filter) {
        if (srcRegion.width == 0) srcRegion.width = std::max(GetDesc(src).extent.width >> srcRegion.mip, 1u);
        if (srcRegion.height == 0) srcRegion.height = std::max(GetDesc(src).extent.height >> srcRegion.mip, 1u);
        if (srcRegion.depth == 0) srcRegion.depth = std::max(GetDesc(src).extent.depth >> srcRegion.mip, 1u);
        if (dstRegion.width == 0) dstRegion.width = std::max(GetDesc(dst).extent.width >> dstRegion.mip, 1u);
        if (dstRegion.height == 0) dstRegion.height = std::max(GetDesc(dst).extent.height >> dstRegion.mip, 1u);
        if (dstRegion.depth == 0) dstRegion.depth = std::max(GetDesc(dst).extent.depth >> dstRegion.mip, 1u);

        VkImageBlit blit = {};
        blit.srcOffsets[0] = {srcRegion.x, srcRegion.y, srcRegion.z};
        blit.srcOffsets[1] = {srcRegion.x + srcRegion.width, srcRegion.y + srcRegion.height, srcRegion.z + srcRegion.depth};
        blit.srcSubresource.aspectMask = DoesFormatHaveDepth(GetDesc(src).format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = srcRegion.mip;
        blit.srcSubresource.baseArrayLayer = srcRegion.layer;
        blit.srcSubresource.layerCount = 1;

        blit.dstOffsets[0] = {dstRegion.x, dstRegion.y, dstRegion.z};
        blit.dstOffsets[1] = {dstRegion.x + dstRegion.width, dstRegion.y + dstRegion.height, dstRegion.z + dstRegion.depth};
        blit.dstSubresource.aspectMask = DoesFormatHaveDepth(GetDesc(dst).format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = dstRegion.mip;
        blit.dstSubresource.baseArrayLayer = dstRegion.layer;
        blit.dstSubresource.layerCount = 1;

        vkCmdBlitImage(GetFrame().cmd, ToInternal(src).image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ToInternal(dst).image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, FILTER_VK[(int)filter]);
    }
    void CmdCopy(Image& src, Image& dst, uint32_t srcMip, uint32_t srcLayer, uint32_t dstMip, uint32_t dstLayer, uint32_t layerCount) {
        EVK_ASSERT(src, "src = null");
        EVK_ASSERT(dst, "dst = null");

        VkImageCopy copy = {};
        copy.extent = {GetDesc(src).extent.width, GetDesc(src).extent.height, GetDesc(src).extent.depth};
        copy.srcOffset = {0, 0, 0};
        copy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.srcSubresource.mipLevel = srcMip;
        copy.srcSubresource.baseArrayLayer = srcLayer;
        copy.srcSubresource.layerCount = layerCount;
        copy.dstOffset = {0, 0, 0};
        copy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.dstSubresource.mipLevel = dstMip;
        copy.dstSubresource.baseArrayLayer = dstLayer;
        copy.dstSubresource.layerCount = layerCount;

        vkCmdCopyImage(GetFrame().cmd, ToInternal(src).image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ToInternal(dst).image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    }
    void CmdCopy(Buffer& src, Image& dst, uint32_t mip, uint32_t layer) {
        auto extent = GetDesc(dst).extent;

        VkBufferImageCopy copy = {};
        copy.bufferOffset = 0;
        copy.bufferRowLength = 0;
        copy.bufferImageHeight = 0;
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.mipLevel = mip;
        copy.imageSubresource.baseArrayLayer = layer;
        copy.imageSubresource.layerCount = 1;
        copy.imageOffset = {0, 0, 0};
        copy.imageExtent = {extent.width >> mip, extent.height >> mip, extent.depth >> mip};

        vkCmdCopyBufferToImage(GetFrame().cmd, ToInternal(src).buffer, ToInternal(dst).image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    }
    void CmdCopy(Buffer& src, Image& dst, const std::vector<ImageRegion>& regions) {
        EVK_ASSERT(regions.size() <= 16, "regions size must be less or equals than 16");

        auto extent = GetDesc(dst).extent;

        VkBufferImageCopy copies[16] = {};
        for (int i = 0; i < regions.size(); i++) {
            VkBufferImageCopy& copy = copies[i];
            const ImageRegion& region = regions[i];

            copy.bufferOffset = 0;
            copy.bufferRowLength = 0;
            copy.bufferImageHeight = 0;
            copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy.imageSubresource.mipLevel = region.mip;
            copy.imageSubresource.baseArrayLayer = region.layer;
            copy.imageSubresource.layerCount = 1;
            copy.imageOffset = {0, 0, 0};
            copy.imageExtent = {extent.width >> region.mip, extent.height >> region.mip, extent.depth >> region.mip};
        }

        vkCmdCopyBufferToImage(GetFrame().cmd, ToInternal(src).buffer, ToInternal(dst).image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, uint32_t(regions.size()), copies);
    }
    void CmdCopy(Buffer& src, Buffer& dst, uint64_t size, uint64_t srcOffset, uint64_t dstOffset) {
        EVK_ASSERT(size > 0, "Size must be bigger than 0");
        EVK_ASSERT((size + dstOffset) <= ToInternal(dst).allocation->GetSize(), "size + dstOffset must be smaller or equal than buffer size");

        VkBufferCopy copy{};
        copy.srcOffset = srcOffset;
        copy.dstOffset = dstOffset;
        copy.size = size;

        vkCmdCopyBuffer(GetFrame().cmd, ToInternal(src).buffer, ToInternal(dst).buffer, 1, &copy);
    }

    void CmdCopy(void* src, Image& dst, uint64_t size, uint32_t mip, uint32_t layer) {
        EVK_ASSERT(size > 0, "Size must be bigger than 0");

        auto& F = GetFrame();
        auto& extent = GetDesc(dst).extent;

        EVK_ASSERT(F.stagingOffset + size < 64'000'000, "Staging buffer out of memory");

        uint64_t staging = uint64_t(F.stagingBuffer.GetPtr()) + F.stagingOffset;
        std::memcpy((void*)staging, src, size);

        VkBufferImageCopy copy = {};
        copy.bufferOffset = F.stagingOffset;
        copy.bufferRowLength = 0;
        copy.bufferImageHeight = 0;
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.mipLevel = mip;
        copy.imageSubresource.baseArrayLayer = layer;
        copy.imageSubresource.layerCount = 1;
        copy.imageOffset = {0, 0, 0};
        copy.imageExtent = {extent.width >> mip, extent.height >> mip, extent.depth >> mip};

        F.stagingOffset += size;

        vkCmdCopyBufferToImage(GetFrame().cmd, ToInternal(F.stagingBuffer).buffer, ToInternal(dst).image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    }
    void CmdCopy(void* src, Buffer& dst, uint64_t size, uint64_t dstOffset) {
        EVK_ASSERT(size > 0, "Size must be bigger than 0");
        // EVK_ASSERT(F.stagingOffset + size < 64'000'000, "Staging buffer out of memory");

        auto& F = GetFrame();
        if (F.stagingOffset + size >= 64'000'000) {
            printf("[evk] [warn] Creating extra staging buffer of size %llu!!! FIXME\n", size);
            Buffer tempStaging = CreateBuffer({
                .size = size,
                .usage = BufferUsage::TransferSrc,
                .memoryType = MemoryType::CPU,
            });
            uint64_t staging = uint64_t(tempStaging.GetPtr());
            std::memcpy((void*)staging, src, size);
            CmdCopy(tempStaging, dst, size, 0u, dstOffset);
        } else {
            uint64_t staging = uint64_t(F.stagingBuffer.GetPtr()) + F.stagingOffset;
            std::memcpy((void*)staging, src, size);
            CmdCopy(F.stagingBuffer, dst, size, F.stagingOffset, dstOffset);
            F.stagingOffset += size;
        }
    }

    void CmdVertex(Buffer& buffer, uint64_t offset) {
        VkDeviceSize rawOffset = offset;
        vkCmdBindVertexBuffers(GetFrame().cmd, 0, 1, &ToInternal(buffer).buffer, &rawOffset);
    }
    void CmdIndex(Buffer& buffer, bool useHalf, uint64_t offset) {
        VkDeviceSize rawOffset = offset;
        vkCmdBindIndexBuffer(GetFrame().cmd, ToInternal(buffer).buffer, rawOffset, useHalf ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
    }
    void CmdBeginRender(Image* attachments, ClearValue* clearValues, int attachmentCount) {
        EVK_ASSERT(GetFrame().insideRenderPass == false, "render pass already bound!");
        EVK_ASSERT(attachments, "attachments = nullptr");
        EVK_ASSERT(attachmentCount > 0, "attachmentCount = 0");
        EVK_ASSERT(attachmentCount < MAX_ATTACHMENTS_COUNT, "Number of attachments %d greater than %d", attachmentCount, MAX_ATTACHMENTS_COUNT);

        GetFrame().insideRenderPass = true;

        bool hasDepth = false;
        bool hasStencil = false;
        uint32_t colorAttachmentCount = 0;
        VkRenderingAttachmentInfoKHR attachInfos[MAX_ATTACHMENTS_COUNT];
        for (int i = 0; i < attachmentCount; i++) {
            auto& desc = GetDesc(attachments[i]);
            auto& attach = attachInfos[i];
            auto& clear = clearValues[i];
            bool isDepthStencil = DoesFormatHaveDepth(desc.format);
            bool isStencil = DoesFormatHaveStencil(desc.format);
            EVK_ASSERT((uint32_t)ToInternal(attachments[i]).desc.usage & (uint32_t)ImageUsage::Attachment, "Image '%s' which is attachment %d don't have ImageUsage::Attachment", GetDesc(attachments[i]).name.c_str(), i);

            attach = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR};
            if (isDepthStencil || isStencil) {
                EVK_ASSERT(i == attachmentCount - 1, "DepthStencil attachment must be in the last attachment index!");
                hasDepth = isDepthStencil;
                hasStencil = isStencil;
            } else {
                colorAttachmentCount++;
            }

            attach.pNext = nullptr;
            attach.imageView = ToInternal(attachments[i]).view;
            attach.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR;
            attach.resolveMode = VK_RESOLVE_MODE_NONE_KHR;
            attach.resolveImageView = VK_NULL_HANDLE;
            attach.resolveImageLayout = {};
            attach.loadOp = clearValues ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            if (clearValues) {
                if (isDepthStencil || isStencil) {
                    VkClearDepthStencilValue clearDepthStencil;
                    clearDepthStencil.depth = clear.depthStencil.depth;
                    clearDepthStencil.stencil = clear.depthStencil.stencil;
                    attach.clearValue.depthStencil = clearDepthStencil;
                } else {
                    VkClearColorValue clearColor;
                    for (int j = 0; j < 4; j++) {
                        clearColor.int32[j] = clear.color.int32[j];
                    }
                    attach.clearValue.color = clearColor;
                }
            }
        }

        Extent extent = GetDesc(attachments[0]).extent;
        VkRenderingInfo info = {VK_STRUCTURE_TYPE_RENDERING_INFO_KHR};
        info.flags = 0;
        info.renderArea = VkRect2D{{0, 0}, {extent.width, extent.height}};
        info.layerCount = 1;
        info.viewMask = 0;
        info.colorAttachmentCount = colorAttachmentCount;
        info.pColorAttachments = attachInfos;
        info.pDepthAttachment = hasDepth ? &attachInfos[attachmentCount - 1] : nullptr;
        info.pStencilAttachment = hasStencil ? &attachInfos[attachmentCount - 1] : nullptr;
        info.pNext = nullptr;

        VkViewport viewport{};
        viewport.x = 0;
        viewport.y = (float)extent.height;
        viewport.width = (float)extent.width;
        viewport.height = -(float)extent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(GetFrame().cmd, 0, 1, &viewport);

        VkRect2D scissor = {};
        scissor.offset.x = 0;
        scissor.offset.y = 0;
        scissor.extent.width = extent.width;
        scissor.extent.height = extent.height;
        vkCmdSetScissor(GetFrame().cmd, 0, 1, &scissor);

        vkCmdSetLineWidth(GetFrame().cmd, 1.0f);

        vkCmdBeginRendering(GetFrame().cmd, &info);
    }
    void CmdEndRender() {
        EVK_ASSERT(GetFrame().insideRenderPass == true, "no render pass bound!");
        GetFrame().insideRenderPass = false;
        vkCmdEndRendering(GetFrame().cmd);
    }
    void CmdBeginPresent() {
        auto& F = GetFrame();
        EVK_ASSERT(!F.doingPresent, "CmdBeginPresent have already been called this frame.");
        F.doingPresent = true;

        auto& S = GetState();
        S.swapchainSemaphoreIndex = (S.swapchainSemaphoreIndex + 1) % S.frames.size();
        VkResult r = vkAcquireNextImageKHR(S.device, S.swapchain, 0, S.frames[S.swapchainSemaphoreIndex].imageReadySemaphore, 0, &S.swapchainIndex);
        CmdBarrier(S.frames[S.swapchainIndex].image, ImageLayout::Undefined, ImageLayout::Attachment);

        ClearValue clears[] = {ClearColor{0.0f, 0.0f, 0.0f, 1.0f}};
        CmdBeginRender(&S.frames[S.swapchainIndex].image, clears, 1);
    }
    void CmdEndPresent() {
        CmdEndRender();
        CmdBarrier(GetState().frames[GetState().swapchainIndex].image, ImageLayout::Attachment, ImageLayout::Present);
    }
    void CmdViewport(float x, float y, float w, float h, float minDepth, float maxDepth) {
        VkViewport viewport = {};
        viewport.x = x;
        viewport.y = y;
        viewport.width = w;
        viewport.height = h;
        viewport.minDepth = minDepth;
        viewport.maxDepth = maxDepth;
        vkCmdSetViewport(GetFrame().cmd, 0, 1, &viewport);
    }
    void CmdScissor(int32_t x, int32_t y, uint32_t w, uint32_t h) {
        VkRect2D scissor = {};
        scissor.offset.x = x;
        scissor.offset.y = y;
        scissor.extent.width = w;
        scissor.extent.height = h;
        vkCmdSetScissor(GetFrame().cmd, 0, 1, &scissor);
    }
    void CmdPush(void* data, uint32_t size, uint32_t offset) {
        EVK_ASSERT(size % 4 == 0, "Push constant 'size' must be aligned by 4 bytes!");
        EVK_ASSERT(offset % 4 == 0, "Push constant 'offset' must be aligned by 4 bytes!");
        EVK_ASSERT(offset + size <= 128, "Push constant offset+size must be smaller than 128 bytes!");
        vkCmdPushConstants(GetFrame().cmd, GetState().pipelineLayout, VK_SHADER_STAGE_ALL, offset, size, data);
    }
    void CmdClear(Image image, ClearValue value) {
        EVK_ASSERT(GetFrame().insideRenderPass == false, "can't be used inside a render pass.");
        CmdBarrier(image, ImageLayout::Undefined, ImageLayout::TransferDst);
        const ImageDesc& desc = GetDesc(image);
        if (DoesFormatHaveDepth(desc.format)) {
            VkClearDepthStencilValue vkValue = {.depth = value.depthStencil.depth, .stencil = value.depthStencil.stencil};
            VkImageSubresourceRange range = {.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, .baseMipLevel = 0, .levelCount = desc.mipCount, .baseArrayLayer = 0, .layerCount = desc.layerCount};
            vkCmdClearDepthStencilImage(GetFrame().cmd, ToInternal(image).image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &vkValue, 1, &range);
        } else {
            VkClearColorValue vkValue = {.uint32 = {value.color.uint32[0], value.color.uint32[1], value.color.uint32[3], value.color.uint32[4]}};
            VkImageSubresourceRange range = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = desc.mipCount, .baseArrayLayer = 0, .layerCount = desc.layerCount};
            vkCmdClearColorImage(GetFrame().cmd, ToInternal(image).image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &vkValue, 1, &range);
        }
    }
    void CmdLineWidth(float width) {
        vkCmdSetLineWidth(GetFrame().cmd, width);
    }
    void CmdDraw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) {
        vkCmdDraw(GetFrame().cmd, vertexCount, instanceCount, firstVertex, firstInstance);
    }
    void CmdDrawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) {
        vkCmdDrawIndexed(GetFrame().cmd, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
    }
    void CmdDrawIndirect(Buffer& buffer, uint64_t offset, uint32_t drawCount, uint32_t stride) {
        vkCmdDrawIndirect(GetFrame().cmd, ToInternal(buffer).buffer, offset, drawCount, stride);
    }
    void CmdDrawIndexedIndirect(Buffer& buffer, uint64_t offset, uint32_t drawCount, uint32_t stride) {
        vkCmdDrawIndexedIndirect(GetFrame().cmd, ToInternal(buffer).buffer, offset, drawCount, stride);
    }
    void CmdDrawIndirectCount(Buffer& buffer, uint64_t offset, Buffer& countBuffer, uint64_t countBufferOffset, uint32_t drawCount, uint32_t stride) {
        vkCmdDrawIndirectCount(GetFrame().cmd, ToInternal(buffer).buffer, offset, ToInternal(countBuffer).buffer, countBufferOffset, drawCount, stride);
    }
    void CmdDrawIndexedIndirectCount(Buffer& buffer, uint64_t offset, Buffer& countBuffer, uint64_t countBufferOffset, uint32_t drawCount, uint32_t stride) {
        vkCmdDrawIndexedIndirectCount(GetFrame().cmd, ToInternal(buffer).buffer, offset, ToInternal(countBuffer).buffer, countBufferOffset, drawCount, stride);
    }
    int CmdBeginTimestamp(const char* name) {
        VkDebugUtilsLabelEXT label = {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
            .pLabelName = name,
        };
        if(GetState().vkCmdBeginDebugUtilsLabelEXT) {
            GetState().vkCmdBeginDebugUtilsLabelEXT(GetFrame().cmd, &label);
        }
        int id = GetFrame().AllocTimestap(name);
        vkCmdWriteTimestamp(GetFrame().cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, GetFrame().queryPool, id * 2);
        return id;
    }
    void CmdEndTimestamp(int id) {
        vkCmdWriteTimestamp(GetFrame().cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, GetFrame().queryPool, id * 2 + 1);
        if(GetState().vkCmdEndDebugUtilsLabelEXT) {
            GetState().vkCmdEndDebugUtilsLabelEXT(GetFrame().cmd);
        }
    }

    BLAS CreateBLAS(const BLASDesc& desc) {
        auto& S = GetState();
        Internal_BLAS* res = new Internal_BLAS();
        res->aabbsBuffer = desc.aabbs;
        res->vertexBuffer = desc.vertices;
        res->indexBuffer = desc.indices;
        if (desc.geometry == GeometryType::Triangles) {
            EVK_ASSERT(desc.stride != 0, "BLASDesc::stride must be different than 0 for triangles");
            EVK_ASSERT(desc.indices, "BLASDesc::indices must not be null");
            EVK_ASSERT(desc.vertices, "BLASDesc::vertices must not be null");
            EVK_ASSERT(desc.triangleCount > 0, "BLASDesc::triangleCount must not be zero");

            VkAccelerationStructureGeometryTrianglesDataKHR triangles = {
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
                .vertexData = {.deviceAddress = ToInternal(desc.vertices).deviceAddress},
                .vertexStride = desc.stride,
                .maxVertex = desc.vertexCount,
                .indexType = VK_INDEX_TYPE_UINT32,
                .indexData = {.deviceAddress = ToInternal(desc.indices).deviceAddress},
                .transformData = {.hostAddress = nullptr},  // identity transform // TODO: support multiple geometries
            };
            res->geometries.push_back(VkAccelerationStructureGeometryKHR{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
                .geometry = {.triangles = triangles},
                .flags = VK_GEOMETRY_OPAQUE_BIT_KHR | VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR,
            });
            VkAccelerationStructureBuildRangeInfoKHR range = {
                .primitiveCount = desc.triangleCount,
                .primitiveOffset = 0,
                .firstVertex = 0,
                .transformOffset = 0,
            };
            res->ranges.push_back(range);
            res->primCounts.push_back(range.primitiveCount);
        } else if (desc.geometry == GeometryType::AABBs) {
            EVK_ASSERT(desc.aabbs, "BLASDesc::aabbs must not be null");
            EVK_ASSERT(desc.aabbsCount, "BLASDesc::aabbsCount must not be zero");

            VkAccelerationStructureGeometryAabbsDataKHR aabbs = {
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR,
                .data = ToInternal(desc.aabbs).deviceAddress,
                .stride = sizeof(VkAabbPositionsKHR),
            };
            res->geometries.push_back(VkAccelerationStructureGeometryKHR{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                .geometryType = VK_GEOMETRY_TYPE_AABBS_KHR,
                .geometry = {.aabbs = aabbs},
                .flags = VK_GEOMETRY_OPAQUE_BIT_KHR | VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR,
            });
            VkAccelerationStructureBuildRangeInfoKHR range = {
                .primitiveCount = desc.aabbsCount,
                .primitiveOffset = 0,
                .firstVertex = 0,
                .transformOffset = 0,
            };
            res->ranges.push_back(range);
            res->primCounts.push_back(desc.aabbsCount);
        }

        res->buildInfo = VkAccelerationStructureBuildGeometryInfoKHR{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR|VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_KHR,
            .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            .geometryCount = static_cast<uint32_t>(res->geometries.size()),
            .pGeometries = res->geometries.data(),
        };

        res->sizeInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        S.vkGetAccelerationStructureBuildSizesKHR(S.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &res->buildInfo, res->primCounts.data(), &res->sizeInfo);

        res->buffer = CreateBuffer({
            .name = "BLAS Buffer",
            .size = res->sizeInfo.accelerationStructureSize,
            .usage = BufferUsage::Storage | BufferUsage::AccelerationStructure,
            .memoryType = MemoryType::GPU,
        });
        VkAccelerationStructureCreateInfoKHR createInfo = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            .buffer = ToInternal(res->buffer).buffer,
            .size = res->sizeInfo.accelerationStructureSize,
            .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        };

        S.vkCreateAccelerationStructureKHR(S.device, &createInfo, nullptr, &res->accel);

        return BLAS{res};
    }
    TLAS CreateTLAS(uint32_t maxBlasCount, bool allowUpdate) {
        auto& S = GetState();
        Internal_TLAS* res = new Internal_TLAS();

        res->instances.resize(maxBlasCount);

        VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
        if (allowUpdate) flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;

        res->instancesBuffer = CreateBuffer({
            .name = "Instances Buffer",
            .size = sizeof(VkAccelerationStructureInstanceKHR) * res->instances.size(),
            .usage = BufferUsage::Storage | BufferUsage::AccelerationStructureInput,
            .memoryType = MemoryType::CPU_TO_GPU,
        });

        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        vkCmdPipelineBarrier(GetFrame().cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        VkAccelerationStructureGeometryInstancesDataKHR instancesVk = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
            .data = {ToInternal(res->instancesBuffer).deviceAddress},
        };
        res->geometry = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
            .geometry = {.instances = instancesVk},
        };

        // Find sizes
        res->buildInfo = VkAccelerationStructureBuildGeometryInfoKHR{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
            .flags = flags,
            .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            .srcAccelerationStructure = VK_NULL_HANDLE,
            .dstAccelerationStructure = VK_NULL_HANDLE,
            .geometryCount = 1,
            .pGeometries = &res->geometry,
        };

        res->sizeInfo = VkAccelerationStructureBuildSizesInfoKHR{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        S.vkGetAccelerationStructureBuildSizesKHR(S.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &res->buildInfo, &maxBlasCount, &res->sizeInfo);

        res->buffer = CreateBuffer({
            .name = "TLAS Buffer",
            .size = res->sizeInfo.accelerationStructureSize,
            .usage = BufferUsage::Storage | BufferUsage::AccelerationStructure,
            .memoryType = MemoryType::GPU,
        });

        VkAccelerationStructureCreateInfoKHR createInfo = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            .buffer = ToInternal(res->buffer).buffer,
            .size = res->sizeInfo.accelerationStructureSize,
            .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        };
        S.vkCreateAccelerationStructureKHR(S.device, &createInfo, nullptr, &res->accel);

        {
            res->resourceid = S.tlasSlots.alloc();
            VkWriteDescriptorSetAccelerationStructureKHR descriptorAccelerationStructure = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
                .accelerationStructureCount = 1,
                .pAccelerationStructures = &res->accel,
            };
            VkWriteDescriptorSet writeDescSet = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = &descriptorAccelerationStructure,
                .dstSet = S.descriptorSet,
                .dstBinding = BINDING_TLAS,
                .dstArrayElement = static_cast<uint32_t>(res->resourceid),
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            };
            vkUpdateDescriptorSets(S.device, 1, &writeDescSet, 0, nullptr);
        }

        return TLAS{res};
    }

    void CmdBuildBLAS(const std::vector<BLAS>& blases, bool update) {
        auto& S = GetState();
        VkDeviceSize batchSize = {0};
        VkDeviceSize batchSizeLimit = {256'000'000};

        VkDeviceSize scratchSize = {0};
        for (auto& blasRes : blases) {
            scratchSize = std::max(scratchSize, ToInternal(blasRes).sizeInfo.buildScratchSize);
        }

        if(scratchSize == 0u)
            return;

        Buffer scratchBuffer = CreateBuffer({
            .name = "BLAS Build Scratch Buffer",
            .size = scratchSize,
            .usage = BufferUsage::Storage,
            .memoryType = MemoryType::GPU,
        });

        std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos = {};
        std::vector<VkAccelerationStructureBuildRangeInfoKHR*> buildRanges = {};

        int i = 0;
        for (auto& blasRes : blases) {
            Internal_BLAS& blas = ToInternal(blasRes);
            batchSize += blas.sizeInfo.accelerationStructureSize;

            EVK_ASSERT(update == false && blas.buildInfo.dstAccelerationStructure == VK_NULL_HANDLE, "BLAS is already built, did you meant to build with update == true?");

            blas.buildInfo.mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            blas.buildInfo.scratchData = {ToInternal(scratchBuffer).deviceAddress};
            blas.buildInfo.dstAccelerationStructure = blas.accel;
            if (update) {
                blas.buildInfo.srcAccelerationStructure = blas.accel;
            }

            buildInfos.push_back(blas.buildInfo);
            buildRanges.push_back(blas.ranges.data());

            auto range = blas.ranges.data();
            S.vkCmdBuildAccelerationStructuresKHR(GetFrame().cmd, 1, &blas.buildInfo, &range);
            VkMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
            };
            vkCmdPipelineBarrier(GetFrame().cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            VkAccelerationStructureDeviceAddressInfoKHR addressInfo = {
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
                .accelerationStructure = blas.accel,
            };
            blas.accStructureDeviceAddress = S.vkGetAccelerationStructureDeviceAddressKHR(S.device, &addressInfo);

            // TODO: Compactation
            // if (batchSize >= batchSizeLimit || i == blases.size() - 1) {
            //    Sync();  // TODO: replace with SubmitSync()
            //    buildInfos.clear();
            //    buildRanges.clear();
            //    batchSize = 0;
            //}
            i++;
            // Cleanup not used resources
            // TODO: if (desc.allowUpdate == false) {
                blas.aabbsBuffer = {};
                blas.indexBuffer = {};
                blas.vertexBuffer = {};
            //}
        }
    }
    void CmdBuildTLAS(const TLAS& tlas, const std::vector<BLASInstance>& blasInstances, bool update) {
        auto& S = GetState();
        Internal_TLAS& res = ToInternal(tlas);

        EVK_ASSERT(tlas, "Invalid TLAS.");
        EVK_ASSERT(blasInstances.size() < res.instances.size(), "TLAS has been created with max of %llu BLAS count but now is being built with %llu BLAS count!",
                    res.instances.size(), blasInstances.size());

        for (int i = 0; i < blasInstances.size(); i++) {
            const BLASInstance& blasInstance = blasInstances[i];
            Internal_BLAS& internalBlas = ToInternal(blasInstance.blas);
            EVK_ASSERT(internalBlas.accStructureDeviceAddress != 0u, "BLAS is not built");

            VkTransformMatrixKHR transform{};
            transform.matrix[0][0] = blasInstance.transform[0];
            transform.matrix[0][1] = blasInstance.transform[1];
            transform.matrix[0][2] = blasInstance.transform[2];
            transform.matrix[0][3] = blasInstance.transform[3];
            transform.matrix[1][0] = blasInstance.transform[4];
            transform.matrix[1][1] = blasInstance.transform[5];
            transform.matrix[1][2] = blasInstance.transform[6];
            transform.matrix[1][3] = blasInstance.transform[7];
            transform.matrix[2][0] = blasInstance.transform[8];
            transform.matrix[2][1] = blasInstance.transform[9];
            transform.matrix[2][2] = blasInstance.transform[10];
            transform.matrix[2][3] = blasInstance.transform[11];

            res.instances[i] = VkAccelerationStructureInstanceKHR{
                .transform = transform,
                .instanceCustomIndex = blasInstance.customId,
                .mask = blasInstance.mask,
                .instanceShaderBindingTableRecordOffset = 0,
                .flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
                .accelerationStructureReference = internalBlas.accStructureDeviceAddress,
            };
        }
        WriteBuffer(res.instancesBuffer, res.instances.data(), blasInstances.size() * sizeof(VkAccelerationStructureInstanceKHR));

        Buffer scratchBuffer = CreateBuffer({
            .name = "TLAS Scratch",
            .size = res.sizeInfo.buildScratchSize,
            .usage = BufferUsage::Storage,
            .memoryType = MemoryType::GPU,
        });

        // Update build information
        res.buildInfo.mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        res.buildInfo.srcAccelerationStructure = res.accel;
        res.buildInfo.dstAccelerationStructure = res.accel;
        res.buildInfo.scratchData.deviceAddress = ToInternal(scratchBuffer).deviceAddress;

        // Build Offsets info: n instances
        VkAccelerationStructureBuildRangeInfoKHR buildOffsetInfo{static_cast<uint32_t>(blasInstances.size()), 0, 0, 0};
        const VkAccelerationStructureBuildRangeInfoKHR* pBuildOffsetInfo = &buildOffsetInfo;

        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        vkCmdPipelineBarrier(GetFrame().cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        // Build the TLAS
        S.vkCmdBuildAccelerationStructuresKHR(GetFrame().cmd, 1, &res.buildInfo, &pBuildOffsetInfo);
    }
}  // namespace evk