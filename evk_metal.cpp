#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#include "evk_metal.h"

// spirv-cross PathTrace.comp.spv --msl --msl-version 20300 --msl-argument-buffer-tier 1

namespace evk {
    Pipeline CreatePipeline(const PipelineDesc& desc){
        return {};
    }
    Buffer CreateBuffer(const BufferDesc& desc){
        return {};
    }
    Image CreateImage(const ImageDesc& desc){
        return {};
    }

    void Resource::decRef() {
        refCount--;
        if (refCount == 0) {
            //GetFrame().toDelete.push_back(this);
        }
    }
    void* Buffer::GetPtr() {}
    void WriteBuffer(Buffer& buffer, void* src, uint64_t size, uint64_t offset){}
    void ReadBuffer(Buffer& buffer, void* dst, uint64_t size, uint64_t offset){}

    RID GetRID(const ResourceRef& ref){}
    const BufferDesc& GetDesc(const Buffer& res){}
    const ImageDesc& GetDesc(const Image& res){}

    static struct Internal {
        MTL::Device* device;
        MTL::CommandQueue* queue;
        CA::MetalLayer* swapchain;
        MTK::View* view;
    } global;
    bool InitializeEVK(const EvkDesc& info) {
        global.device = MTL::CreateSystemDefaultDevice();
        global.queue = global.device->newCommandQueue();
        global.swapchain = CA::MetalLayer::layer();
        global.swapchain->setDevice(global.device);
        
        CGRect frame = (CGRect){{100.0, 100.0}, {512.0, 512.0}};
        global.view = MTK::View::alloc()->init(frame, global.device);
        global.view->setColorPixelFormat(MTL::PixelFormat::PixelFormatBGRA8Unorm_sRGB);
        global.view->setClearColor(MTL::ClearColor::Make( 1.0, 0.0, 0.0, 1.0 ));

        return true;
    }
    void Shutdown(){}
    bool InitializeSwapchain(void* nsWindow){
        NS::Window* window = (NS::Window*)nsWindow;
        // NS::Private::Selector::s_ksetContentView_
        window->contentView()->setLayer(global.swapchain);
        // NS::Private::Selector::s_ksetContentView_

        return true;
    }

    uint32_t GetFrameBufferingCount(){}
    uint32_t GetFrameIndex(){}
    void Submit(){
        MTL::ClearColor color = MTL::ClearColor::Make(0, 0, 255, 1);
        CA::MetalDrawable* surface = global.swapchain->nextDrawable();
        
        MTL::RenderPassDescriptor* pass = MTL::RenderPassDescriptor::renderPassDescriptor()->autorelease();
        pass->colorAttachments()->object(0)->setClearColor(color);
        pass->colorAttachments()->object(0)->setLoadAction(MTL::LoadActionClear);
        pass->colorAttachments()->object(0)->setStoreAction(MTL::StoreActionStore);
        pass->colorAttachments()->object(0)->setTexture(surface->texture());

        MTL::CommandBuffer* cmd = global.queue->commandBuffer()->autorelease();
        MTL::RenderCommandEncoder* encoder = cmd->renderCommandEncoder(pass)->autorelease();
        encoder->endEncoding();
        cmd->presentDrawable(surface);
        cmd->commit();
    }
    void Sync(){}
    const std::vector<TimestampEntry>& GetTimestamps(){
        static std::vector<TimestampEntry> ts = {};
        return ts;
    }
    void CmdBind(Pipeline pipeline){}
    void CmdDispatch(uint32_t countX, uint32_t countY, uint32_t countZ){}
    void CmdBarrier(Image& image, ImageLayout oldLayout, ImageLayout newLayout, uint32_t mip, uint32_t mipCount, uint32_t layer, uint32_t layerCount){}
    void CmdBarrier(){}
    void CmdFill(Buffer dst, uint32_t data, uint64_t size, uint64_t offset){}
    void CmdUpdate(Buffer& dst, uint64_t dstOffset, uint64_t size, void* src){}
    void CmdBlit(Image& src, Image& dst, ImageRegion srcRegion, ImageRegion dstRegion, Filter filter){}
    void CmdCopy(Image& src, Image& dst, uint32_t srcMip, uint32_t srcLayer, uint32_t dstMip, uint32_t dstLayer, uint32_t layerCount){}
    void CmdCopy(Buffer& src, Image& dst, uint32_t mip, uint32_t layer){}
    void CmdCopy(Buffer& src, Image& dst, const std::vector<ImageRegion>& regions){}
    void CmdCopy(Buffer& src, Buffer& dst, uint64_t size, uint64_t srcOffset, uint64_t dstOffset){}

    void CmdCopy(void* src, Image& dst, uint64_t size, uint32_t mip, uint32_t layer){}
    void CmdCopy(void* src, Buffer& dst, uint64_t size, uint64_t dstOffset){}

    void CmdVertex(Buffer& buffer, uint64_t offset){}
    void CmdIndex(Buffer& buffer, bool useHalf, uint64_t offset){}

    void CmdBeginRender(Image* attachments, ClearValue* clearValues, int attachmentCount){}
    void CmdEndRender(){}
    void CmdBeginPresent(){}
    void CmdEndPresent(){}
    void CmdViewport(float x, float y, float w, float h, float minDepth, float maxDepth){}
    void CmdScissor(int32_t x, int32_t y, uint32_t w, uint32_t h){}
    void CmdPush(void* data, uint32_t size, uint32_t offset){}
    void CmdClear(Image image, ClearValue value){}
    void CmdDraw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance){}
    void CmdDrawIndexed(uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance){}
    void CmdDrawIndirect(Buffer& buffer, uint64_t offset, uint32_t drawCount, uint32_t stride){}
    void CmdDrawIndexedIndirect(Buffer& buffer, uint64_t offset, uint32_t drawCount, uint32_t stride){}
    void CmdDrawIndirectCount(Buffer& buffer, uint64_t offset, Buffer& countBuffer, uint64_t countBufferOffset, uint32_t drawCount, uint32_t stride){}
    void CmdDrawIndexedIndirectCount(Buffer& buffer, uint64_t offset, Buffer& countBuffer, uint64_t countBufferOffset, uint32_t drawCount, uint32_t stride){}
    void CmdBeginTimestamp(const char* name){}
    void CmdEndTimestamp(const char* name){}

    namespace rt {
        BLAS CreateBLAS(const BLASDesc& desc){}
        TLAS CreateTLAS(uint64_t blasCount, bool allowUpdate){}

        void CmdBuildBLAS(const std::vector<BLAS>& blases, bool update){}
        void CmdBuildTLAS(const TLAS& tlas, const std::vector<BLASInstance>& blasInstances, bool update){}
    }
}

