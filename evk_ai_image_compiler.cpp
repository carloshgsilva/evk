#include "evk_ai_ir.h"

#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace evk::ai::detail {
std::vector<uint8_t> load_embedded_shader(std::string_view name);
}

namespace evk::ai {
namespace {

struct ImageInputs {
    evk::RID color;
    evk::RID albedo;
    evk::RID normal;
    evk::RID unused = 0;
};
static_assert(sizeof(ImageInputs) == 16u);

std::unique_ptr<Tensor> pack_bias(const TensorData& source,
                                  uint32_t packed_channels) {
    uint32_t channels = source.desc.shape[0];
    if (packed_channels < channels) {
        throw std::runtime_error("packed bias storage is smaller than its logical channels");
    }
    auto packed = std::make_unique<Tensor>(Shape({16u, packed_channels}));
    float16_t* destination = packed->cpu();
    const float16_t* source_data = source.values.data();
    std::fill(destination, destination + packed->shape.count(), float16_t(0.0f));
    for (uint32_t row = 0; row < 16u; ++row) {
        std::copy(source_data, source_data + channels,
                  destination + row * packed_channels);
    }
    packed->cpu_upload(false);
    return packed;
}

uint32_t collapsed_tap(uint32_t phase, uint32_t kernel_index) {
    if (phase == 0u) return kernel_index == 0u ? 0u : 1u;
    return kernel_index == 2u ? 1u : 0u;
}

std::unique_ptr<Tensor> pack_3x3_weight(const TensorData& source,
                                        uint32_t skip_channels,
                                        uint32_t packed_output_channels) {
    uint32_t output_channels = source.desc.shape[0];
    uint32_t input_channels = source.desc.shape[1];
    constexpr uint32_t kernel_elements = 9u;

    bool compact_rgb_input = input_channels == 3u && skip_channels == 0u;
    bool compact_rgb_skip = skip_channels == 3u;

    uint32_t low_channels = 0u;
    if (skip_channels != 0u) {
        if (input_channels <= skip_channels ||
            ((input_channels - skip_channels) % 16u) != 0u) {
            throw std::runtime_error("unsupported collapsed phase channel count");
        }
        low_channels = input_channels - skip_channels;
    }

    if (packed_output_channels < output_channels) {
        throw std::runtime_error("packed weight storage is smaller than its logical channels");
    }
    uint32_t packed_k = ((input_channels + 15u) / 16u) * kernel_elements * 16u;
    if (compact_rgb_input) {
        packed_k = kernel_elements * 8u;
    } else if (compact_rgb_skip) {
        packed_k = low_channels / 16u * kernel_elements * 16u +
                   kernel_elements * 8u;
    }
    uint32_t phase_k = low_channels * 4u;
    auto packed = std::make_unique<Tensor>(
        Shape({packed_k + 4u * phase_k, packed_output_channels}));
    float16_t* destination = packed->cpu();
    const float16_t* source_data = source.values.data();
    std::fill(destination, destination + packed->shape.count(), float16_t(0.0f));

    for (uint32_t output_channel = 0; output_channel < output_channels;
         ++output_channel) {
        for (uint32_t input_channel = 0; input_channel < input_channels;
             ++input_channel) {
            for (uint32_t tap = 0; tap < kernel_elements; ++tap) {
                uint32_t k;
                if (compact_rgb_input) {
                    k = tap * 8u + input_channel;
                } else if (compact_rgb_skip && input_channel >= low_channels) {
                    k = low_channels / 16u * kernel_elements * 16u +
                        tap * 8u + input_channel - low_channels;
                } else {
                    k = ((input_channel / 16u) * kernel_elements + tap) * 16u +
                        input_channel % 16u;
                }
                uint32_t source_index =
                    (output_channel * input_channels + input_channel) *
                        kernel_elements + tap;
                destination[k * packed_output_channels + output_channel] =
                    source_data[source_index];
            }
        }
    }

    if (skip_channels != 0u) {
        for (uint32_t phase_y = 0u; phase_y < 2u; ++phase_y) {
            for (uint32_t phase_x = 0u; phase_x < 2u; ++phase_x) {
                uint32_t phase = phase_y * 2u + phase_x;
                for (uint32_t output_channel = 0u;
                     output_channel < output_channels; ++output_channel) {
                    for (uint32_t input_channel = 0u;
                         input_channel < low_channels; ++input_channel) {
                        for (uint32_t tap_y = 0u; tap_y < 2u; ++tap_y) {
                            for (uint32_t tap_x = 0u; tap_x < 2u; ++tap_x) {
                                float sum = 0.0f;
                                for (uint32_t kernel_y = 0u; kernel_y < 3u;
                                     ++kernel_y) {
                                    if (collapsed_tap(phase_y, kernel_y) != tap_y) {
                                        continue;
                                    }
                                    for (uint32_t kernel_x = 0u; kernel_x < 3u;
                                         ++kernel_x) {
                                        if (collapsed_tap(phase_x, kernel_x) != tap_x) {
                                            continue;
                                        }
                                        uint32_t source_index =
                                            ((output_channel * input_channels +
                                              input_channel) * 3u + kernel_y) *
                                                3u + kernel_x;
                                        sum += float(source_data[source_index]);
                                        // Fast decoder weights must match the
                                        // runtime FP16 fragment-add order.
                                        if (!compact_rgb_skip) {
                                            sum = float(float16_t(sum));
                                        }
                                    }
                                }
                                uint32_t tap = tap_y * 2u + tap_x;
                                uint32_t k =
                                    (input_channel / 16u * 4u + tap) * 16u +
                                    input_channel % 16u;
                                destination[(packed_k + phase * phase_k + k) *
                                                packed_output_channels +
                                            output_channel] = float16_t(sum);
                            }
                        }
                    }
                }
            }
        }
    }

    packed->cpu_upload(false);
    return packed;
}

struct KernelCatalog {
    struct ConvKernel {
        uint32_t input_channels;
        uint32_t output_channels;
        uint32_t rows = 64u;
        uint32_t width_multiple = 1u;
        evk::Pipeline pipeline;
    };

    struct PhaseKernel {
        uint32_t low_channels;
        uint32_t skip_channels;
        uint32_t output_channels;
        uint32_t rows;
        evk::Pipeline regular;
        evk::Pipeline tail;
    };

    evk::Pipeline conv;
    evk::Pipeline narrow_conv;
    evk::Pipeline tail_conv;
    evk::Pipeline narrow_tail_conv;
    evk::Pipeline phase_concat_conv;
    evk::Pipeline narrow_phase_concat_conv;
    evk::Pipeline prepacked_phase_concat_conv;
    evk::Pipeline phase_tail_concat_conv;
    evk::Pipeline narrow_phase_tail_concat_conv;
    evk::Pipeline prepacked_phase_tail_concat_conv;
    evk::Pipeline conv_final;
    evk::Pipeline conv_final_rows;
    evk::Pipeline conv_pool;
    std::array<ConvKernel, 3> pool_kernels;
    evk::Pipeline conv_32_rows;
    evk::Pipeline conv_64_32_rows;
    evk::Pipeline conv_64_64_rows;
    evk::Pipeline from_rgba_image;
    evk::Pipeline to_rgb;
    evk::Pipeline to_rgba_image;
    std::array<ConvKernel, 12> conv_kernels;
    std::array<PhaseKernel, 7> phase_kernels;

    KernelCatalog() {
        conv = create("oidn_conv");
        narrow_conv = create("oidn_narrow_conv", "oidn_conv_narrow");
        tail_conv = create("oidn_tail_conv", "oidn_conv",
                           evk::Constant{0u, 0u, 0u, 0u, 1u});
        narrow_tail_conv = create("oidn_narrow_tail_conv", "oidn_conv_narrow",
                                  evk::Constant{0u, 0u, 0u, 0u, 1u});
        phase_concat_conv = create("oidn_concat_phase");
        narrow_phase_concat_conv = create("oidn_concat_phase_narrow");
        prepacked_phase_concat_conv = create("oidn_concat_phase_prepacked");
        phase_tail_concat_conv = create(
            "oidn_phase_tail_concat_conv", "oidn_concat_phase",
            evk::Constant{0u, 0u, 0u, 1u});
        narrow_phase_tail_concat_conv = create(
            "oidn_narrow_phase_tail_concat_conv", "oidn_concat_phase_narrow",
            evk::Constant{0u, 0u, 0u, 1u});
        prepacked_phase_tail_concat_conv = create(
            "oidn_prepacked_phase_tail_concat_conv", "oidn_concat_phase_prepacked",
            evk::Constant{0u, 0u, 0u, 1u});
        conv_final = create("oidn_conv_output_8");
        conv_final_rows = create("oidn_conv_output_8_rows");
        conv_pool = create("oidn_conv_pool");
        pool_kernels = {{
            {32u, 48u, 64u, 1u, create_conv_pool("oidn_balanced_conv_pool_32_48", 32u, 48u)},
            {48u, 64u, 64u, 1u, create_conv_pool("oidn_balanced_conv_pool_48_64", 48u, 64u)},
            {64u, 80u, 64u, 1u, create_conv_pool("oidn_balanced_conv_pool_64_80", 64u, 80u)},
        }};
        conv_32_rows = create("oidn_conv_32_rows");
        conv_64_32_rows = create_decoder_rows("oidn_conv_64_32_rows", 32u);
        conv_64_64_rows = create_decoder_rows("oidn_conv_64_64_rows", 64u);
        from_rgba_image = create("oidn_from_rgba_image");
        to_rgb = create("oidn_to_rgb");
        to_rgba_image = create("oidn_to_rgba_image");
        conv_kernels = {{
            {3u, 32u, 64u, 1u, create_regular("oidn_fast_conv_3_32", 3u, 32u, "oidn_conv_input_8")},
            {32u, 32u, 64u, 1u, create_regular("oidn_fast_conv_32_32", 32u, 32u)},
            {64u, 64u, 64u, 1u, create_regular("oidn_fast_conv_64_64", 64u, 64u)},
            {64u, 32u, 64u, 1u, create_regular("oidn_fast_conv_64_32", 64u, 32u)},
            {32u, 48u, 64u, 1u, create_regular("oidn_balanced_conv_32_48", 32u, 48u)},
            {48u, 64u, 64u, 1u, create_regular("oidn_balanced_conv_48_64", 48u, 64u)},
            {64u, 80u, 64u, 1u, create_regular("oidn_balanced_conv_64_80", 64u, 80u)},
            {80u, 96u, 64u, 1u, create_regular("oidn_balanced_conv_80_96", 80u, 96u)},
            {96u, 96u, 96u, 96u, create_regular("oidn_balanced_conv_96_96_rows96", 96u, 96u, "oidn_conv_wide_rows96")},
            {96u, 96u, 64u, 1u, create_regular("oidn_balanced_conv_96_96", 96u, 96u)},
            {112u, 112u, 80u, 80u,
             create_regular("oidn_balanced_conv_112_112_rows80", 112u, 112u,
                            "oidn_conv_wide_rows80")},
            {112u, 112u, 64u, 1u, create_regular("oidn_balanced_conv_112_112", 112u, 112u)},
        }};
        phase_kernels = {{
            create_phase_kernel("oidn_fast_phase_32_3_32",
                                "oidn_concat_phase_prepacked", 32u, 3u, 32u),
            create_phase_kernel("oidn_fast_phase_32_32_64",
                                "oidn_concat_phase_collapsed", 32u, 32u, 64u),
            create_phase_kernel("oidn_fast_phase_64_32_64",
                                "oidn_concat_phase_collapsed", 64u, 32u, 64u),
            create_phase_kernel("oidn_balanced_phase_96_64_112",
                                "oidn_concat_phase_collapsed_wide",
                                96u, 64u, 112u, 64u, true),
            create_phase_kernel("oidn_balanced_phase_112_48_96_rows80",
                                "oidn_concat_phase_wide_rows80",
                                112u, 48u, 96u, 80u),
            create_phase_kernel("oidn_balanced_phase_96_32_64_rows96",
                                "oidn_concat_phase_rows96",
                                96u, 32u, 64u, 96u),
            create_phase_kernel("oidn_balanced_phase_64_3_64_rows96",
                                "oidn_concat_phase_prepacked_rows96",
                                64u, 3u, 64u, 96u),
        }};
    }

    static evk::Pipeline create(const char* name, const char* shader = nullptr,
                                evk::ConstantRaw constants = {}) {
        if (!shader) shader = name;
        return evk::CreatePipeline({
            .name = name,
            .CS = evk::ai::detail::load_embedded_shader(shader),
            .constants = constants,
        });
    }

    static evk::Pipeline create_regular(const char* name, uint32_t input_channels,
                                        uint32_t output_channels,
                                        const char* shader = nullptr) {
        if (!shader) {
            shader = output_channels <= 64u ? "oidn_conv_narrow" : "oidn_conv";
        }
        return create(name, shader, evk::Constant{
            0u, input_channels, output_channels,
            (input_channels + 15u) / 16u});
    }

    static evk::Pipeline create_decoder_rows(const char* name,
                                             uint32_t output_channels) {
        return create(name, "oidn_conv_decoder_rows128",
                      evk::Constant{64u, output_channels, 4u});
    }

    static evk::Pipeline create_conv_pool(const char* name,
                                          uint32_t input_channels,
                                          uint32_t output_channels) {
        uint32_t channels_per_workgroup = output_channels == 48u ? 48u : 64u;
        return create(name, "oidn_conv_pool_balanced", evk::Constant{
            input_channels, output_channels, input_channels / 16u,
            channels_per_workgroup, channels_per_workgroup * 4u});
    }

    static evk::Pipeline create_phase(const char* name, const char* shader,
                                      uint32_t low_channels, uint32_t input_channels,
                                      uint32_t output_channels, bool tail = false) {
        return create(name, shader, evk::Constant{
            low_channels / 16u, (input_channels + 15u) / 16u,
            output_channels, uint32_t(tail)});
    }

    static PhaseKernel create_phase_kernel(
        const char* name, const char* shader, uint32_t low_channels,
        uint32_t skip_channels, uint32_t output_channels, uint32_t rows = 64u,
        bool always_tail = false) {
        evk::Pipeline regular = create_phase(
            name, shader, low_channels, low_channels + skip_channels,
            output_channels, always_tail);
        if (always_tail) {
            return {low_channels, skip_channels, output_channels, rows,
                    std::move(regular), {}};
        }
        std::string tail_name = std::string(name) + "_tail";
        return {low_channels, skip_channels, output_channels, rows,
                std::move(regular),
                create_phase(tail_name.c_str(), shader, low_channels,
                             low_channels + skip_channels, output_channels, true)};
    }

    void convolution_rows(const evk::Pipeline& pipeline, evk::Buffer& input,
                          Tensor& weight, Tensor& bias, Tensor& output,
                          uint32_t width, uint32_t height,
                          uint32_t tile_width, uint32_t output_rows) const {
        auto& cmd = evk::ai::GetCmd();
        cmd.bind(pipeline);
        cmd.push(evk::Constant{
            input.GetReference(),
            weight.buffer.GetReference(),
            bias.buffer.GetReference(),
            output.buffer.GetReference(),
            width,
            height,
        });
        cmd.dispatch((width + tile_width - 1u) / tile_width,
                     height / output_rows, 1u);
        cmd.barrier();
    }

    void dispatch_convolution(const evk::Pipeline& pipeline, evk::Buffer& input,
                              Tensor& weight, Tensor& bias, Tensor& output,
                              uint32_t width, uint32_t height,
                              uint32_t input_channels, uint32_t padded_k,
                              uint32_t groups_x, uint32_t groups_y,
                              uint32_t groups_z) const {
        auto& cmd = evk::ai::GetCmd();
        cmd.bind(pipeline);
        cmd.push(evk::Constant{
            input.GetReference(),
            input.GetReference(),
            weight.buffer.GetReference(),
            bias.buffer.GetReference(),
            output.buffer.GetReference(),
            width,
            height,
            input_channels,
            output.shape[3],
            padded_k,
            width,
            input_channels,
            0u,
        });
        cmd.dispatch(groups_x, groups_y, groups_z);
        cmd.barrier();
    }

    void convolution(Tensor& input, Tensor& weight, Tensor& bias, Tensor& output,
                     bool activation) const {
        uint32_t height = input.shape[1];
        uint32_t width = input.shape[2];
        uint32_t input_channels = input.shape[3];
        uint32_t output_channels = output.shape[3];
        if (!activation) {
            if (input_channels == 32u && output_channels == 8u &&
                (height % 4u) == 0u) {
                convolution_rows(conv_final_rows, input.buffer, weight, bias,
                                 output, width, height, 64u, 4u);
                return;
            }
            dispatch_convolution(conv_final, input.buffer, weight, bias, output,
                                 width, height, input_channels, weight.shape[0],
                                 (width + 63u) / 64u, 1u, height);
            return;
        }

        cooperative_convolution(input.buffer, weight, bias, output,
                                width, height, input_channels);
    }

    void concat_convolution(Tensor& low_resolution, Tensor& skip, Tensor& weight,
                            Tensor& bias, Tensor& output) const {
        uint32_t output_channels = output.shape[3];
        // Each matrix row writes 16 pixels with a stride of two.
        bool phase_tail = (skip.shape[2] % 32u) != 0u;
        const evk::Pipeline* pipeline = nullptr;
        uint32_t rows = 64u;
        for (const PhaseKernel& kernel : phase_kernels) {
            if (kernel.low_channels == low_resolution.shape[3] &&
                kernel.skip_channels == skip.shape[3] &&
                kernel.output_channels == output_channels) {
                pipeline = phase_tail && kernel.tail
                    ? &kernel.tail : &kernel.regular;
                rows = kernel.rows;
                break;
            }
        }
        if (!pipeline) {
            if (skip.shape[3] == 3u) {
                pipeline = phase_tail ? &prepacked_phase_tail_concat_conv
                                      : &prepacked_phase_concat_conv;
            } else if (output_channels <= 64u) {
                pipeline = phase_tail ? &narrow_phase_tail_concat_conv
                                      : &narrow_phase_concat_conv;
            } else {
                pipeline = phase_tail ? &phase_tail_concat_conv
                                      : &phase_concat_conv;
            }
        }

        auto& cmd = evk::ai::GetCmd();
        cmd.bind(*pipeline);
        cmd.push(evk::Constant{
            low_resolution.buffer.GetReference(),
            skip.buffer.GetReference(),
            weight.buffer.GetReference(),
            bias.buffer.GetReference(),
            output.buffer.GetReference(),
            skip.shape[2],
            skip.shape[1],
            low_resolution.shape[3] + skip.shape[3],
            output_channels,
            ((low_resolution.shape[3] + skip.shape[3] + 15u) / 16u) * 9u * 16u,
            low_resolution.shape[2],
            low_resolution.shape[3],
            skip.shape[3],
        });
        uint32_t phase_pixels = (skip.shape[2] + 1u) / 2u;
        cmd.dispatch((phase_pixels + rows - 1u) / rows, skip.shape[1], 2u);
        cmd.barrier();
    }

    void cooperative_convolution(evk::Buffer& input, Tensor& weight,
                                 Tensor& bias, Tensor& output,
                                 uint32_t width, uint32_t height,
                                 uint32_t input_channels) const {
        uint32_t output_channels = output.shape[3];
        bool tail = (width % 16u) != 0u;
        if (!tail && input_channels == 64u && output_channels == 32u &&
            (height % 4u) == 0u) {
            convolution_rows(conv_64_32_rows, input, weight, bias, output,
                             width, height, 128u, 4u);
            return;
        }
        if (!tail && input_channels == 64u && output_channels == 64u &&
            (height % 2u) == 0u) {
            convolution_rows(conv_64_64_rows, input, weight, bias, output,
                             width, height, 128u, 2u);
            return;
        }
        if (!tail && input_channels == 32u && output_channels == 32u &&
            (height % 4u) == 0u) {
            convolution_rows(conv_32_rows, input, weight, bias, output,
                             width, height, 64u, 4u);
            return;
        }
        bool narrow = output_channels <= 64u;
        const evk::Pipeline* specialized = nullptr;
        uint32_t rows = 64u;
        if (!tail) {
            for (const ConvKernel& kernel : conv_kernels) {
                if (kernel.input_channels == input_channels &&
                    kernel.output_channels == output_channels &&
                    (width % kernel.width_multiple) == 0u) {
                    specialized = &kernel.pipeline;
                    rows = kernel.rows;
                    break;
                }
            }
        }
        const evk::Pipeline& pipeline = tail
            ? (narrow ? narrow_tail_conv : tail_conv)
            : (specialized ? *specialized : (narrow ? narrow_conv : conv));
        uint32_t channels_per_workgroup = narrow ? 64u : 128u;
        dispatch_convolution(
            pipeline, input, weight, bias, output, width, height, input_channels,
            ((input_channels + 15u) / 16u) * 9u * 16u,
            (width + rows - 1u) / rows,
            (output_channels + channels_per_workgroup - 1u) /
                channels_per_workgroup,
            height);
    }

    void convolution_pooling(Tensor& input, Tensor& weight, Tensor& bias,
                             Tensor& output) const {
        auto& cmd = evk::ai::GetCmd();
        const evk::Pipeline* pipeline = &conv_pool;
        for (const ConvKernel& kernel : pool_kernels) {
            if (kernel.input_channels == input.shape[3] &&
                kernel.output_channels == output.shape[3]) {
                pipeline = &kernel.pipeline;
                break;
            }
        }
        cmd.bind(*pipeline);
        cmd.push(evk::Constant{
            input.buffer.GetReference(),
            weight.buffer.GetReference(),
            bias.buffer.GetReference(),
            output.buffer.GetReference(),
            input.shape[2],
            input.shape[1],
        });
        uint32_t output_channels = output.shape[3];
        uint32_t group_channels = output_channels == 64u ? 64u : 48u;
        cmd.dispatch((input.shape[2] + 63u) / 64u,
                     input.shape[1] / (output_channels == 32u ? 4u : 2u),
                     (output_channels + group_channels - 1u) / group_channels);
        cmd.barrier();
    }

    void convert_from_rgba(ImageInputs inputs, Tensor& output, uint32_t width,
                           uint32_t height, uint32_t padded_width,
                           uint32_t padded_height, uint32_t channels) const {
        auto& cmd = evk::ai::GetCmd();
        cmd.bind(from_rgba_image);
        cmd.push(evk::Constant{
            inputs,
            output.buffer.GetReference(),
            width,
            height,
            padded_width,
            padded_height,
            channels,
        });
        uint32_t pixels = padded_width * padded_height;
        cmd.dispatch((pixels + 255u) / 256u, 1u, 1u);
        cmd.barrier();
    }

    void convert_to_rgb(Tensor& input, evk::Buffer& output, uint32_t width,
                        uint32_t height, uint32_t padded_width) const {
        auto& cmd = evk::ai::GetCmd();
        cmd.bind(to_rgb);
        cmd.push(evk::Constant{
            input.buffer.GetReference(),
            output.GetReference(),
            width,
            height,
            padded_width,
            input.shape[3],
        });
        uint32_t elements = width * height * 3u;
        cmd.dispatch((elements + 255u) / 256u, 1u, 1u);
        cmd.barrier();
    }

    void convert_to_rgba(Tensor& input, evk::Image& output, uint32_t width,
                         uint32_t height, uint32_t padded_width) const {
        auto& cmd = evk::ai::GetCmd();
        cmd.bind(to_rgba_image);
        cmd.push(evk::Constant{
            input.buffer.GetReference(),
            output.GetRID(),
            width,
            height,
            padded_width,
            input.shape[3],
        });
        uint32_t pixels = width * height;
        cmd.dispatch((pixels + 255u) / 256u, 1u, 1u);
        cmd.barrier();
    }
};

NodeId convolution_node(const Graph& graph, const PlanStep& step) {
    for (NodeId id : step.nodes) {
        if (std::holds_alternative<Conv2D>(graph.node(id).operation)) return id;
    }
    return invalid_node;
}

std::pair<ValueId, ValueId> phase_inputs(const Graph& graph, const Node& conv) {
    ValueId low_resolution;
    ValueId skip;
    for (ValueId candidate : graph.node(conv.inputs[0].node).inputs) {
        const Node& node = graph.node(candidate.node);
        if (std::holds_alternative<Upsample2D>(node.operation)) {
            low_resolution = node.inputs[0];
        } else {
            skip = candidate;
        }
    }
    return {low_resolution, skip};
}

Tensor& materialized(const std::vector<Tensor*>& values, ValueId value) {
    Tensor* tensor = values[value.node];
    if (!tensor) {
        throw std::runtime_error("evk::ai compiler referenced an unmaterialized value");
    }
    return *tensor;
}

KernelCatalog& image_kernels(std::optional<KernelCatalog>& kernels) {
    if (!kernels) {
        throw std::runtime_error("image adapters require an NHWC execution plan");
    }
    return *kernels;
}

bool supports_image_plan(const Graph& graph, const Plan& plan) {
    if (graph.inputs().size() != 1u || graph.outputs().size() != 1u ||
        !evk::GetFeatures().coopmat) {
        return false;
    }
    const TensorDesc& input = graph.desc(graph.inputs()[0]);
    if (input.type != DataType::Float16 || input.shape.rank() != 4u ||
        input.shape[0] != 1u) {
        return false;
    }

    for (const PlanStep& step : plan) {
        switch (step.kernel) {
        case Kernel::Conv2D:
        case Kernel::Conv2DRelu:
        case Kernel::Conv2DReluMaxPool2D:
        case Kernel::Upsample2DConcatConv2DRelu:
            break;
        default:
            return false;
        }

        NodeId conv_id = convolution_node(graph, step);
        if (conv_id == invalid_node) return false;
        const Node& conv = graph.node(conv_id);
        const Conv2D& attributes = std::get<Conv2D>(conv.operation);
        const Shape& weight = graph.desc(conv.inputs[1]).shape;
        if (attributes.stride_y != 1u || attributes.stride_x != 1u ||
            attributes.pad_y != 1u || attributes.pad_x != 1u ||
            attributes.groups != 1u || weight[2] != 3u || weight[3] != 3u) {
            return false;
        }
        if (!std::holds_alternative<Constant>(
                graph.node(conv.inputs[1].node).operation) ||
            !std::holds_alternative<Constant>(
                graph.node(conv.inputs[2].node).operation)) {
            return false;
        }
        if (step.kernel == Kernel::Conv2D &&
            (graph.outputs()[0].node != step.nodes.back() ||
             graph.node(step.nodes.back()).result.shape[1] > 8u)) {
            return false;
        }
        if (step.kernel == Kernel::Upsample2DConcatConv2DRelu) {
            if ((graph.desc(phase_inputs(graph, conv).first).shape[1] % 16u) != 0u) {
                return false;
            }
        }
    }
    return true;
}

} // namespace


struct Binding {
    Tensor* tensor;
    Layout layout;
};

struct Executable::Impl {
    std::optional<KernelCatalog> kernels;
    ::Graph runtime_graph;
    std::vector<Binding> inputs;
    std::vector<Binding> outputs;
    Plan fusion_plan;
};

Executable::Executable() = default;
Executable::~Executable() = default;
Executable::Executable(Executable&&) noexcept = default;
Executable& Executable::operator=(Executable&&) noexcept = default;

Executable::Executable(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Tensor& Executable::input(uint32_t index) {
    if (!impl_) throw std::runtime_error("evk::ai executable is empty");
    Impl& impl = *impl_;
    if (index >= impl.inputs.size()) {
        throw std::runtime_error("evk::ai executable input index is out of range");
    }
    return *impl.inputs[index].tensor;
}

Tensor& Executable::output(uint32_t index) {
    if (!impl_) throw std::runtime_error("evk::ai executable is empty");
    Impl& impl = *impl_;
    if (index >= impl.outputs.size()) {
        throw std::runtime_error("evk::ai executable output index is out of range");
    }
    return *impl.outputs[index].tensor;
}

Layout Executable::input_layout(uint32_t index) const {
    if (!impl_) throw std::runtime_error("evk::ai executable is empty");
    if (index >= impl_->inputs.size()) {
        throw std::runtime_error("evk::ai executable input index is out of range");
    }
    return impl_->inputs[index].layout;
}

Layout Executable::output_layout(uint32_t index) const {
    if (!impl_) throw std::runtime_error("evk::ai executable is empty");
    if (index >= impl_->outputs.size()) {
        throw std::runtime_error("evk::ai executable output index is out of range");
    }
    return impl_->outputs[index].layout;
}

const Plan& Executable::plan() const {
    if (!impl_) throw std::runtime_error("evk::ai executable is empty");
    return impl_->fusion_plan;
}

void Executable::eval(bool submit, bool wait, bool profile) {
    if (!impl_) throw std::runtime_error("evk::ai executable is empty");
    impl_->runtime_graph.eval(false, submit, wait, profile);
}

void Executable::from_rgba(evk::Image& color, uint32_t width, uint32_t height,
                           uint32_t padded_width, uint32_t padded_height) {
    if (!impl_) throw std::runtime_error("evk::ai executable is empty");
    Impl& impl = *impl_;
    image_kernels(impl.kernels).convert_from_rgba(
        {color.GetRID(), color.GetRID(), color.GetRID()},
        *impl.inputs[0].tensor, width, height, padded_width, padded_height, 3u);
}

void Executable::from_rgba(evk::Image& color, evk::Image& albedo,
                           evk::Image& normal, uint32_t width, uint32_t height,
                           uint32_t padded_width, uint32_t padded_height) {
    if (!impl_) throw std::runtime_error("evk::ai executable is empty");
    Impl& impl = *impl_;
    image_kernels(impl.kernels).convert_from_rgba(
        {color.GetRID(), albedo.GetRID(), normal.GetRID()},
        *impl.inputs[0].tensor, width, height, padded_width, padded_height, 9u);
}

void Executable::to_rgb(evk::Buffer& output_buffer, uint32_t width,
                        uint32_t height, uint32_t padded_width) {
    if (!impl_) throw std::runtime_error("evk::ai executable is empty");
    Impl& impl = *impl_;
    image_kernels(impl.kernels).convert_to_rgb(
        *impl.outputs[0].tensor, output_buffer, width, height, padded_width);
}

void Executable::to_rgba(evk::Image& output_image, uint32_t width,
                         uint32_t height, uint32_t padded_width) {
    if (!impl_) throw std::runtime_error("evk::ai executable is empty");
    Impl& impl = *impl_;
    image_kernels(impl.kernels).convert_to_rgba(
        *impl.outputs[0].tensor, output_image, width, height, padded_width);
}

Executable compile(const Graph& semantic, const DataStore& data) {
    semantic.validate(data);

    auto impl = std::make_unique<Executable::Impl>();
    Plan optimized_plan = build_fusion_plan(semantic);
    if (!supports_image_plan(semantic, optimized_plan)) {
        std::vector<Tensor*> values(semantic.nodes().size(), nullptr);
        bool has_pending_uploads = false;

        for (NodeId id = 0; id < semantic.nodes().size(); ++id) {
            const Node& node = semantic.node(id);
            const TensorDesc& result = node.result;

            if (std::holds_alternative<Input>(node.operation)) {
                Tensor& tensor = impl->runtime_graph.tensor(result.shape);
                tensor.name = node.debug_name;
                values[id] = &tensor;
                impl->inputs.push_back({&tensor, Layout::NCHW});
                continue;
            }

            const TensorData* source_data = nullptr;
            if (const auto* constant = std::get_if<Constant>(&node.operation)) {
                source_data = &data.get(constant->data);
            } else if (const auto* parameter =
                           std::get_if<Parameter>(&node.operation)) {
                source_data = &data.get(parameter->data);
            }
            if (source_data) {
                Tensor& tensor = impl->runtime_graph.tensor(result.shape);
                std::copy(source_data->values.begin(), source_data->values.end(),
                          tensor.cpu());
                tensor.cpu_upload(false);
                has_pending_uploads = true;
                tensor.name = node.debug_name;
                values[id] = &tensor;
                continue;
            }

            Tensor* output = nullptr;
            Kernel kernel;
            if (const auto* conv = std::get_if<Conv2D>(&node.operation)) {
                if (conv->stride_y != conv->stride_x ||
                    conv->pad_y != conv->pad_x || conv->groups != 1u) {
                    throw std::runtime_error(
                        "generic Conv2D currently requires symmetric stride/padding and one group");
                }
                output = &impl->runtime_graph.conv2d(
                    materialized(values, node.inputs[0]),
                    materialized(values, node.inputs[1]),
                    materialized(values, node.inputs[2]),
                    conv->stride_y, conv->pad_y);
                kernel = Kernel::Conv2D;
            } else if (std::holds_alternative<Relu>(node.operation)) {
                output = &impl->runtime_graph.relu(materialized(values, node.inputs[0]));
                kernel = Kernel::Relu;
            } else if (const auto* pool = std::get_if<MaxPool2D>(&node.operation)) {
                if (pool->kernel_y != pool->kernel_x ||
                    pool->stride_y != pool->stride_x) {
                    throw std::runtime_error(
                        "generic MaxPool2D currently requires square symmetric attributes");
                }
                output = &impl->runtime_graph.max_pool2d(
                    materialized(values, node.inputs[0]),
                    pool->kernel_y, pool->stride_y);
                kernel = Kernel::MaxPool2D;
            } else if (const auto* upsample =
                           std::get_if<Upsample2D>(&node.operation)) {
                if (upsample->scale_y != upsample->scale_x) {
                    throw std::runtime_error(
                        "generic Upsample2D currently supports symmetric nearest scaling");
                }
                output = &impl->runtime_graph.upsample2d(
                    materialized(values, node.inputs[0]), upsample->scale_y);
                kernel = Kernel::Upsample2D;
            } else if (const auto* concat = std::get_if<Concat>(&node.operation)) {
                if (concat->axis != 1) {
                    throw std::runtime_error(
                        "generic Concat currently supports the NCHW channel axis");
                }
                output = &impl->runtime_graph.concat(
                    materialized(values, node.inputs[0]),
                    materialized(values, node.inputs[1]));
                kernel = Kernel::Concat;
            } else {
                throw std::runtime_error("evk::ai fallback operation is unsupported");
            }

            output->name = node.debug_name;
            values[id] = output;
            impl->fusion_plan.push_back(PlanStep{kernel, {id}});
        }

        for (ValueId output : semantic.outputs()) {
            impl->outputs.push_back({&materialized(values, output), Layout::NCHW});
        }
        if (has_pending_uploads) SubmitCmd(true);
        return Executable(std::move(impl));
    }

    impl->fusion_plan = std::move(optimized_plan);
    KernelCatalog* kernels = &impl->kernels.emplace();
    const TensorDesc& input_desc = semantic.desc(semantic.inputs()[0]);
    Tensor& input = impl->runtime_graph.tensor(Shape({
        input_desc.shape[0], input_desc.shape[2],
        input_desc.shape[3], input_desc.shape[1]
    }));
    impl->inputs.push_back({&input, Layout::NHWC});

    std::vector<Tensor*> values(semantic.nodes().size(), nullptr);
    values[semantic.inputs()[0].node] = &input;

    auto logical_data = [&semantic, &data](ValueId value) -> const TensorData& {
        const Node& node = semantic.node(value.node);
        return data.get(std::get<Constant>(node.operation).data);
    };

    for (const PlanStep& step : impl->fusion_plan) {
        NodeId conv_id = convolution_node(semantic, step);
        const Node& conv = semantic.node(conv_id);
        NodeId root = step.nodes.back();
        const TensorDesc& result = semantic.node(root).result;
        bool final_convolution = step.kernel == Kernel::Conv2D;
        uint32_t storage_channels = final_convolution ? 8u : result.shape[1];
        uint32_t packed_output_channels = final_convolution
            ? 8u : (result.shape[1] + 15u) & ~15u;

        ValueId low_resolution;
        ValueId skip;
        uint32_t skip_channels = 0u;
        if (step.kernel == Kernel::Upsample2DConcatConv2DRelu) {
            auto inputs = phase_inputs(semantic, conv);
            low_resolution = inputs.first;
            skip = inputs.second;
            skip_channels = semantic.desc(skip).shape[1];
        }

        Tensor& weight = *impl->runtime_graph.nodes.emplace_back(
            pack_3x3_weight(logical_data(conv.inputs[1]), skip_channels,
                            packed_output_channels));
        Tensor& bias = *impl->runtime_graph.nodes.emplace_back(
            pack_bias(logical_data(conv.inputs[2]), packed_output_channels));

        Tensor& output = impl->runtime_graph.tensor(Shape({
            result.shape[0], result.shape[2], result.shape[3], storage_channels
        }));
        output.name = conv.debug_name;

        if (step.kernel == Kernel::Conv2DReluMaxPool2D) {
            Tensor& source = materialized(values, conv.inputs[0]);
            output.forward_fn = [kernels, &source, &weight, &bias, &output]() {
                kernels->convolution_pooling(source, weight, bias, output);
            };
        } else if (step.kernel == Kernel::Upsample2DConcatConv2DRelu) {
            Tensor& low_tensor = materialized(values, low_resolution);
            Tensor& skip_tensor = materialized(values, skip);
            output.forward_fn = [kernels, &low_tensor, &skip_tensor,
                                 &weight, &bias, &output]() {
                kernels->concat_convolution(
                    low_tensor, skip_tensor, weight, bias, output);
            };
        } else if (step.kernel == Kernel::Conv2DRelu) {
            Tensor& source = materialized(values, conv.inputs[0]);
            output.forward_fn = [kernels, &source, &weight, &bias, &output]() {
                kernels->convolution(source, weight, bias, output, true);
            };
        } else if (step.kernel == Kernel::Conv2D) {
            Tensor& source = materialized(values, conv.inputs[0]);
            output.forward_fn = [kernels, &source, &weight, &bias, &output]() {
                kernels->convolution(source, weight, bias, output, false);
            };
        } else {
            throw std::runtime_error("invalid built-in image kernel step");
        }
        values[root] = &output;
    }

    impl->outputs.push_back(
        {&materialized(values, semantic.outputs()[0]), Layout::NHWC});
    SubmitCmd(true);
    return Executable(std::move(impl));
}

} // namespace evk::ai
