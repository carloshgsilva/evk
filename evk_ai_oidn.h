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

// OIDN's RT LDR U-Net loaded from an official .tza weights archive.
class Model {
public:
    explicit Model(const std::string& weights_path);
    ~Model();

    bool loaded() const { return !parameters_.empty(); }
    uint32_t input_channels() const { return input_channels_; }

    Tensor& build(Graph& graph, Tensor& input) const;
    void convert_from_rgba(evk::Image& color, Tensor& output, uint32_t width,
                           uint32_t height, uint32_t padded_width,
                           uint32_t padded_height) const;
    void convert_from_rgba(evk::Image& color, evk::Image& albedo,
                           evk::Image& normal, Tensor& output, uint32_t width,
                           uint32_t height, uint32_t padded_width,
                           uint32_t padded_height) const;
    void convert_to_rgb(Tensor& input, evk::Buffer& output, uint32_t width,
                        uint32_t height, uint32_t padded_width) const;
    void convert_to_rgba(Tensor& input, evk::Image& output, uint32_t width,
                         uint32_t height, uint32_t padded_width) const;

private:
    struct Kernels;

    void load(const std::string& weights_path);
    Tensor& parameter(const std::string& name) const;
    Tensor& packed_parameter(const std::string& name) const;

    std::unordered_map<std::string, std::unique_ptr<Tensor>> parameters_;
    std::unordered_map<std::string, std::unique_ptr<Tensor>> packed_parameters_;
    std::unique_ptr<Kernels> kernels_;
    uint32_t input_channels_ = 0;
};

// Reusable LDR denoiser. The loaded weights determine whether the model accepts
// color only or color with albedo and normal auxiliary inputs.
class Denoiser {
public:
    Denoiser(const std::string& weights_path, uint32_t width, uint32_t height);

    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    bool uses_auxiliary_inputs() const { return model_.input_channels() == 9u; }

    void denoise(std::span<const float> color_rgb, std::span<float> output_rgb,
                 bool profile = false);
    void denoise(std::span<const float> color_rgb,
                 std::span<const float> albedo_rgb,
                 std::span<const float> normal_xyz,
                 std::span<float> output_rgb, bool profile = false);

    // Records denoising between RGBA8Unorm or RGBA16Sfloat storage images
    // without submitting or waiting. Images must match the denoiser dimensions
    // and already be in the General layout. Output alpha is set to one.
    void denoise(evk::Cmd& cmd, evk::Image& input_rgba, evk::Image& output_rgba);

    // Records denoising with auxiliary albedo and normal images. This requires
    // rt_ldr_alb_nrm weights. Color and albedo are in [0, 1]; normals are
    // signed world-space or view-space vectors in [-1, 1].
    void denoise(evk::Cmd& cmd, evk::Image& color_rgba,
                 evk::Image& albedo_rgba, evk::Image& normal_rgba,
                 evk::Image& output_rgba);

    const std::vector<evk::TimestampEntry>& timings() const { return timings_; }
    const CpuTimings& cpu_timings() const { return cpu_timings_; }

private:
    void denoise_cpu(std::span<const float> color_rgb,
                     std::span<const float> albedo_rgb,
                     std::span<const float> normal_xyz,
                     std::span<float> output_rgb, bool profile);

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
