#pragma once

#include "evk_ai.h"

#include <memory>
#include <span>
#include <string>
#include <unordered_map>

namespace evk::ai::oidn {

struct CpuTimings {
    double input_pack_ms = 0.0;
    double graph_ms = 0.0;
    double download_ms = 0.0;
    double output_unpack_ms = 0.0;
};

// OIDN's balanced RT LDR U-Net loaded from an official .tza weights archive.
// Input and output tensors are NHWC FP16 and contain sRGB values in [0, 1].
class Model {
public:
    explicit Model(const std::string& weights_path);
    ~Model();

    bool loaded() const { return !parameters_.empty(); }

    Tensor& build(Graph& graph, Tensor& input) const;
    void convert_to_rgb(Tensor& input, evk::Buffer& output, uint32_t width,
                        uint32_t height, uint32_t padded_width) const;

private:
    struct Kernels;

    void load(const std::string& weights_path);
    Tensor& parameter(const std::string& name) const;
    Tensor& packed_parameter(const std::string& name) const;

    std::unordered_map<std::string, std::unique_ptr<Tensor>> parameters_;
    std::unordered_map<std::string, std::unique_ptr<Tensor>> packed_parameters_;
    std::unique_ptr<Kernels> kernels_;
};

// Reusable color-only LDR denoiser. The public image layout is interleaved RGB;
// conversion to the model's channel-last layout is handled internally.
class Denoiser {
public:
    Denoiser(const std::string& weights_path, uint32_t width, uint32_t height);

    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }

    void denoise(std::span<const float> input_rgb, std::span<float> output_rgb,
                 bool profile = false);
    const std::vector<evk::TimestampEntry>& timings() const { return timings_; }
    const CpuTimings& cpu_timings() const { return cpu_timings_; }

private:
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t padded_width_ = 0;
    uint32_t padded_height_ = 0;
    Model model_;
    Graph graph_;
    Tensor* input_ = nullptr;
    Tensor* output_ = nullptr;
    evk::Buffer output_gpu_;
    evk::Buffer output_cpu_;
    std::vector<evk::TimestampEntry> timings_;
    CpuTimings cpu_timings_;
};

} // namespace evk::ai::oidn
