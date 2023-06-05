#include "evk.h"
#include "evk_internal.h"

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "AppKit/AppKit.hpp"
#include "MetalKit/MetalKit.hpp"
#include "QuartzCore/QuartzCore.hpp"

namespace evk {
    struct State {
        MTL::Device* device;
        MTL::CommandQueue* queue;
        CA::MetalLayer* swapchain;
        MTK::View* view;

        MTL::Buffer* buffer_pos;
        MTL::Buffer* buffer_color;
        MTL::RenderPipelineState* pipeline;
    };
    State& GetState();

    struct PipelineMetal : Resource {
        MTL::RenderPipelineState* renderPipeline;
        MTL::ComputePipelineState* computePipeline;
    };
}